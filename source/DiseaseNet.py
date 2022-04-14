import collections
import gzip
import itertools
import logging
import mmap
import os
import os.path as osp
import sys
import torch

from shutil import copyfile

from sklearn import neighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class DiseaseNet(InMemoryDataset):
    class RawFileEnum:
        disease_hpo = 0
        disease_publication_titles = 1
        all_diseases = 2
        disease_pathway = 3

    class ProcessedFileEnum:
        disease_id_feature_index_mapping = 0
        edges = 1
        nodes = 2
        data = 3

    def __init__(
            self,
            root,
            transform=None,
            pre_transform=None,
            edge_source='databases',
            hpo_count_freq_cutoff=70,
            feature_source='disease_publications',
            skip_truncated_svd=False,
            svd_components=2048,
            svd_n_iter=12,
            n_neighbours=32
    ):
        """

        Args:
            root:
            transform:
            pre_transform:
            edge_source (str): Out of {'databases', 'feature_similarity}. 'databases' will use shared database features
                such as shared pathway etc. 'feature_similarity' will use a kNN approach to create disease links.
            hpo_count_freq_cutoff (int): Consider only disease ontology terms associated to less than
                hpo_count_freq_cutoff diseases for building edges.
            feature_source (list): List of which sources to use to create the disease feature vectors.
                Out of {'disease_publications', 'phenotypes'}
            n_neighbours (int): It the edge_source is set to feature_similarity: Number of most similar nodes to
            consider.
        """
        self.skip_truncated_svd = skip_truncated_svd
        self.svd_components = svd_components
        self.svd_n_iter = svd_n_iter
        self.edge_source = edge_source
        self.feature_source = feature_source
        self.hpo_count_freq_cutoff = hpo_count_freq_cutoff
        self.n_neighbours = n_neighbours
        super(DiseaseNet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.ProcessedFileEnum.data])

    @property
    def raw_file_names(self):
        return [
            'disease_hpo.tsv',
            'disease_publication_titles_and_abstracts.tsv',
            'all_diseases.tsv',
            'disease_pathway.tsv'
        ]

    @property
    def processed_file_names(self):
        return [
            'disease_id_feature_index_mapping.txt',
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
        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.disease_id_feature_index_mapping]):
            logging.info('Create disease_id feature_index mapping.')
            self.create_disease_index_feature_mapping()

        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.nodes]):
            logging.info('Create feature matrix.')
            self.generate_disease_feature_matrix()

        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.edges]):
            logging.info('Create edges.')
            if self.edge_source == 'databases':
                self.generate_edges()
            if self.edge_source == 'feature_similarity':
                self.generate_edges_similarity_based()

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
        x = self.load_node_feature_martix()
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

    def load_node_feature_martix(self):
        return torch.load(self.processed_paths[self.ProcessedFileEnum.nodes])

    def generate_disease_feature_matrix(self):
        logging.info('Creating disease feature vectors.')

        disease_index_mapping = self.load_disease_index_feature_mapping()
        x = None
        if 'phenotypes' in self.feature_source:
            logging.info('Create phenotype feature vectors.')
            mlb = MultiLabelBinarizer()
            hpo_ids = set()
            disease_hpo_map = collections.defaultdict(set)
            with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.disease_hpo])) as disease_hpo_file:
                for disease_link in disease_hpo_file.readlines():
                    dis_id, hpo_id, hpo_name = [s.strip() for s in disease_link.split('\t')]
                    hpo_ids.add(hpo_id)
                    disease_hpo_map[dis_id].add(hpo_id)

            mlb.fit([hpo_ids])
            disease_id_sorted_by_index = sorted(disease_index_mapping.keys(), key=lambda x: disease_index_mapping[x])
            disease_features = [disease_hpo_map[d_id] for d_id in disease_id_sorted_by_index]
            # Create the feature matrix
            x = torch.tensor(mlb.transform(disease_features), dtype=torch.float)

        if 'disease_publications' in self.feature_source:
            logging.info('Create publication feature vectors.')
            disease_id_publication_titles = collections.defaultdict(str)
            corpus = []
            # Start by retrieving the available titles.
            with open(
                self.raw_paths[self.RawFileEnum.disease_publication_titles],
                mode='r',
                encoding='utf-8'
            ) as disease_publications:
                for line in disease_publications:
                    disease_id, publication_title, publication_abstract = [s.strip() for s in line.split('\t')]
                    if len(publication_abstract) > 0:
                        corpus.append(publication_abstract)
                        disease_id_publication_titles[disease_id] += f' {publication_abstract}'
                    if len(publication_title) > 0:
                        corpus.append(publication_title)
                        disease_id_publication_titles[disease_id] += f' {publication_title}'

            # Build the vectorizer
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_df=0.01,  # occurs in max 10% of the diseases.
                min_df=0.001  # occurs in at least 0.1% of the diseases.
            )
            vectorizer.fit(corpus)

            # Create the feature matrix
            tmp = torch.tensor(vectorizer.transform(
                [disease_id_publication_titles[oid] for oid in
                 sorted(disease_index_mapping.keys(), key=lambda x: disease_index_mapping[x])]
            ).toarray(), dtype=torch.float)
            # Concat with previous feature matrix.
            if x is not None:
                x = torch.cat((x, tmp), dim=1)
            else:
                x = tmp

        if not self.skip_truncated_svd:
            logging.info('Doing dimensionality reduction using TruncatedSVD')
            svd = TruncatedSVD(n_components=self.svd_components, n_iter=self.svd_n_iter, random_state=42)
            svd.fit(x)
            x = svd.transform(x)
            x = torch.tensor(x).float()

        torch.save(x, self.processed_paths[self.ProcessedFileEnum.nodes])

    def load_edges(self):
        return torch.load(self.processed_paths[self.ProcessedFileEnum.edges])

    def generate_edges(self):
        disease_index_mapping = self.load_disease_index_feature_mapping()
        to_be_linked_diseases = collections.defaultdict(set)

        logging.info('Generating the disease edges.')
        logging.info('Using shared phenotypes.')
        with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.disease_hpo])) as disease_hpo_file:
            for disease_link in disease_hpo_file.readlines():
                dis_id, hpo_id, hpo_name = [s.strip() for s in disease_link.split('\t')]
                to_be_linked_diseases[hpo_id].add(disease_index_mapping[dis_id])

        logging.info('Using shared pathways.')
        with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.disease_pathway])) as file:
            for pathway_link in file.readlines():
                dis_id, pathway_id = [s.strip() for s in pathway_link.split('\t')]
                to_be_linked_diseases[pathway_id].add(disease_index_mapping[dis_id])

        edges = set()
        len_counts = collections.defaultdict(int)
        for diseases in to_be_linked_diseases.values():
            len_counts[len(diseases)] += 1
            if len(diseases) > self.hpo_count_freq_cutoff:
                continue
            for source, target in itertools.combinations(diseases, 2):
                edges.add((source, target))

        sources, targets, scores = [], [], []
        for source, target in edges:
            scores.append(1)
            sources.append(source)
            targets.append(target)

        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(scores).reshape((len(sources), 1))

        torch.save((edge_index, edge_attr), self.processed_paths[self.ProcessedFileEnum.edges])

    def generate_edges_similarity_based(self):
        logging.info("Generate similarity based edges.")
        X = self.load_node_feature_martix()
        X = normalize(X)  # Normalize so we can use euclidian distance and get the same result as when using cosine sim.
        adj = neighbors.kneighbors_graph(
            X,
            n_neighbors=self.n_neighbours,
            include_self=True
        ).toarray()
        edge_index, edge_attr = dense_to_sparse(torch.tensor(adj))

        torch.save((edge_index, edge_attr), self.processed_paths[self.ProcessedFileEnum.edges])

    def create_disease_index_feature_mapping(self):
        """ Creates a mapping between disease and index to be used in the feature matrix.
        Stores the result to disease_id_feature_index_mapping

        """
        disease_index_mapping = collections.OrderedDict()
        with open(self.raw_paths[self.RawFileEnum.all_diseases], mode='r') as in_file:
            for line in in_file:
                parts = [s.strip() for s in line.split('\t')]
                identifier = parts[1]
                if identifier.startswith('OMIM') and identifier not in disease_index_mapping:
                    disease_index_mapping[identifier] = len(disease_index_mapping)

        with open(self.processed_paths[self.ProcessedFileEnum.disease_id_feature_index_mapping], mode='w') as out_file:
            out_file.write('{disease_id}\t{index}\n')
            for gene_id, index in disease_index_mapping.items():
                out_file.write(f'{gene_id}\t{index}\n')

    def load_disease_index_feature_mapping(self):
        disease_index_mapping = collections.OrderedDict()
        with open(self.processed_paths[self.ProcessedFileEnum.disease_id_feature_index_mapping], mode='r') as file:
            next(file)
            for line in file.readlines():
                disease_id, index = [s.strip() for s in line.split('\t')]
                index = int(index)
                disease_index_mapping[disease_id] = index

        return disease_index_mapping


if __name__ == '__main__':
    HERE = osp.abspath(osp.dirname(__file__))
    DATASET_ROOT = osp.join(HERE, 'data_sources', 'dataset_diseases')

    disease_net = DiseaseNet(
        root=DATASET_ROOT,
        edge_source='feature_similarity',
        feature_source='disease_publications'
    )
    print(disease_net)
