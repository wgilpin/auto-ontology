import os
import sys
import argparse
from tika import parser
from sentences import split
from deep_latent import DeepLatentCluster
from deep_cluster import DeepCluster


def useful(s: str):
    """Filter out sentences that are not useful"""
    return len(s) > 2 and sum(c.isalpha() for c in s) > 2


class FileProcessor():
    """
    Process a folder of files, extracting text and splitting into sentences.
    """

    def __init__(self, input: str, output: str, approach: str, clusterer: str):
        self.folder = input
        self.output = output
        self.approach = approach
        self.clusterer = clusterer

    def process_file(self, file: str):
        """Process a single file"""

        # extract text with tika
        raw = parser.from_file(file)

        # split article into sentences
        sents = split(raw['content'])

        # drop if too short
        drop_short = [p for p in sents if useful(p)]

        print(f"{len(drop_short)} in {file}")
        return drop_short

    def process_folder(self):
        """Process all files in a folder"""

        lines = []
        for file in os.listdir(self.folder):
            try:
                ls = self.process_file(os.path.join(self.folder, file))  # type: ignore
                lines.extend(ls)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        if len(lines) == 0:
            print("No lines found")
            return

        fn = f"{self.output}/lines.txt"
        with open(fn, 'w', encoding='utf-8') as the_file:
            the_file.writelines(f'{s}\n' for s in lines)
        print(f"wrote {len(lines)} total to {fn}")

    def train_evaluate(self, lines_folder, approach, clusterer):
        """Train and evaluate a model on the processed data"""
        print(f"from{lines_folder}: {approach} {clusterer}")

        if approach == 'deep_latent':
            dc = DeepLatentCluster(
            'test-latent-all',
            {
                "cluster": clusterer,
                "folder": lines_folder,
            })
            dc.make_model()
            dc.train_model()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cluster terms in files')
    parser.add_argument('--source', '-s', type=str,
                        help='Folder containing files', default='pdfs')
    parser.add_argument('--output', '-o', type=str,
                        help='Folder where output is stored', default='src/pdfs')
    parser.add_argument('--filename', '-f', type=str,
                        help='output filename, for having multiple experiments',
                        default='lines.txt')
    parser.add_argument('--approach', '-a', choices=['deep_cluster', 'deep_latent'],
                        help='Use Deep Clustering or learn the Latent Representation')
    parser.add_argument('--clusterer', '-c',
                        choices=['kmeans', 'optics', 'agg', 'gmm'],
                        default='kmeans',
                        help='Clusterer to use: K-means, OPTICS, '
                             'Gaussian Mixture Model or agglomerative')
    arguments, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print(f"Unknown arguments: {unknown}")
        parser.print_help()
        sys.exit(1)
    return arguments


if __name__ == "__main__":
    args = parse_args()
    # get folder from command line
    processor = FileProcessor(args.source, args.output, args.approach, args.clusterer)
    processor.process_folder()
    processor.train_evaluate(args.output, args.approach, args.clusterer)
