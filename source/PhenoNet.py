import collections
import gzip
import itertools
import logging
import mmap
import os
import os.path as osp
import sys
import torch

import networkx as nx

from shutil import copyfile

from sklearn import neighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
import pronto

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class PhenoNet(InMemoryDataset):
    class RawFileEnum:
        pheno_hpo = 0
        disease_hpo = 1

    class ProcessedFileEnum:
        pheno_id_feature_index_mapping = 0
        edges = 1
        nodes = 2
        data = 3

    def __init__(
            self,
            root,
            transform=None,
            pre_transform=None,
            edge_source='diseases',
            feature_source='diseases'
    ):
        """

        Args:
            feature_source (list): List of which sources to use to create the phenotype feature vectors.
                Currently only out of {'diseases'}
        """
        self.feature_source = feature_source
        self.edge_source = edge_source
        super(PhenoNet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.ProcessedFileEnum.data])

    @property
    def raw_file_names(self):
        return [
            'hp.obo',
            'phenotype.hpoa'
        ]

    @property
    def processed_file_names(self):
        return [
            'pheno_id_feature_index_mapping.txt',
            'edges.pt',
            'nodes.pt',
            'data.pt'
        ]

    def download(self):
        for file in self.raw_file_names:
            dest = os.path.join(self.raw_dir, file)
            if not os.path.isfile(dest):
                src = os.path.join(self.raw_dir, '..', '..', file)
                copyfile(src, dest)

    def process(self):
        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.pheno_id_feature_index_mapping]):
            logging.info('Create pheno_id feature_index mapping.')
            self.create_pheno_index_feature_mapping()

        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.nodes]):
            logging.info('Create feature matrix.')
            self.generate_pheno_feature_matrix()

        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.edges]):
            logging.info('Create edges.')
            self.generate_edges()

        # Create and store the data object
        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.data]):
            self.generate_data_object()

    @staticmethod
    def get_len_file(in_file, ignore_count=0):
        """ Count the number of lines in a file.

        Args:
            in_file (str):
            ignore_count (int): Remove this from the total count (Ignore headers for example).

        Returns (int): The number of lines in file_path
        """
        fp = open(in_file, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines - ignore_count

    def generate_data_object(self):
        x = self.load_node_feature_matrix()
        edge_index, edge_attr = self.load_edges()
        data_list = [
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logging.info('Storing the data.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.ProcessedFileEnum.data])
        logging.info('Done.')

    def load_node_feature_matrix(self):
        return torch.load(self.processed_paths[self.ProcessedFileEnum.nodes])

    def generate_pheno_feature_matrix(self):
        logging.info('Creating phenotype feature vectors.')

        pheno_index_mapping = self.load_pheno_index_feature_mapping()
        x = None
        if 'diseases' in self.feature_source:
            logging.info('Creating phenotype features based on disease.')
            mlb = MultiLabelBinarizer()
            hpo_ids = set()
            hpo_disease_map = collections.defaultdict(set)
            with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.disease_hpo]), 'r') as disease_hpo_file:
                for line in disease_hpo_file:
                    if line.startswith('#') or line.startswith("database_id"): # header, comments and versioning
                        continue
                    omim_id, name, _, hpo_id, _, _, _, frequency, *_ = line.strip().split('\t')
                    hpo_ids.add(hpo_id)
                    hpo_disease_map[hpo_id].add(omim_id) # include frequency?

            mlb.fit([hpo_ids])
            pheno_id_sorted_by_index = sorted(pheno_index_mapping.keys(), key=lambda x: pheno_index_mapping[x])
            pheno_features = [hpo_disease_map[d_id] for d_id in pheno_id_sorted_by_index]
            # Create the feature matrix
            x = torch.tensor(mlb.transform(pheno_features), dtype=torch.float)

        torch.save(x, self.processed_paths[self.ProcessedFileEnum.nodes])

    def load_edges(self):
        return torch.load(self.processed_paths[self.ProcessedFileEnum.edges])

    def generate_edges(self):
        pheno_index_mapping = self.load_pheno_index_feature_mapping()
        to_be_linked_phenotypes = collections.defaultdict(set)

        logging.info('Generating the phenotype edges.')
        logging.info('Using ontology.')
        pheno_hpo_file = osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.pheno_hpo])
        ontology = pronto.Ontology(pheno_hpo_file)

        edges = set()
        sources, targets, scores = [], [], []
        for term in ontology.terms():
            target = pheno_index_mapping[term.id]
            for parent in term.superclasses(distance=1):
                source = pheno_index_mapping[parent.id]
                edges.add((source, target))
                sources.append(source)
                targets.append(target)
                scores.append(1) # weights?
        
        if 'diseases' in self.edge_source: # might add noise
            disease_hpo_map = collections.defaultdict(set)
            logging.info('Using shared diseases.')
            with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.disease_hpo]), 'r') as disease_hpo_file:
                for line in disease_hpo_file:
                    if line.startswith('#') or line.startswith("database_id"): # header, comments and versioning
                        continue
                    omim_id, name, _, hpo_id, _, _, _, frequency, *_ = line.strip().split('\t')
                    if frequency is not None:
                        frequency = float(frequency.split('/')[0])/float(frequency.split('/')[1])
                    disease_hpo_map[omim_id].add((hpo_id, frequency))
                for id_freq in disease_hpo_map.values():
                    for pair1, pair2 in itertools.combinations(id_freq, 2):
                        source = pheno_index_mapping[pair1[0]]
                        target = pheno_index_mapping[pair2[0]]
                        freq1 = pair1[1]
                        freq2 = pair2[1]
                        sources.append(source)
                        targets.append(target)
                        scores.append(freq1 * freq2) # multiply frequencies (still between 0 and 1)

        print(source)            
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(scores).reshape((len(sources), 1))

        torch.save((edge_index, edge_attr), self.processed_paths[self.ProcessedFileEnum.edges])

    def create_pheno_index_feature_mapping(self):
        """ Creates a mapping between phenotype and index to be used in the feature matrix.
        Stores the result to disease_id_feature_index_mapping

        """
        pheno_index_mapping = collections.OrderedDict()
        ontology = pronto.Ontology(self.raw_paths[self.RawFileEnum.pheno_hpo])
        for term in ontology.terms():
            identifier = term.id 
            if identifier not in pheno_index_mapping:
                pheno_index_mapping[identifier] = len(pheno_index_mapping)

        with open(self.processed_paths[self.ProcessedFileEnum.pheno_id_feature_index_mapping], mode='w') as out_file:
            out_file.write('{pheno_id}\t{index}\n')
            for gene_id, index in pheno_index_mapping.items():
                out_file.write(f'{gene_id}\t{index}\n')

    def load_pheno_index_feature_mapping(self):
        pheno_index_mapping = collections.OrderedDict()
        with open(self.processed_paths[self.ProcessedFileEnum.pheno_id_feature_index_mapping], mode='r') as file:
            next(file)
            for line in file.readlines():
                pheno_id, index = [s.strip() for s in line.split('\t')]
                index = int(index)
                pheno_index_mapping[pheno_id] = index

        return pheno_index_mapping


if __name__ == '__main__':
    HERE = osp.abspath(osp.dirname(__file__))
    DATASET_ROOT = osp.join(HERE, 'data_sources', 'dataset_diseases') # TODO

    pheno_net = PhenoNet(
        root=DATASET_ROOT,
        edge_source='',
        feature_source='diseases'
    )
    print(pheno_net)
