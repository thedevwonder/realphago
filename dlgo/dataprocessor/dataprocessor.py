import time
from dlgo.gosgf import Sgf_game
from dlgo.goboard import Board, GameState, Player, Point, Move
from dlgo.encoders.base import get_encoder_by_name
import numpy as np
import shutil
import tarfile
import gzip
import os
import tqdm


__all__ = [
    'DataProcessor'
]

transformations = [
    'identity',
    'rotate_90',       
    'rotate_180',
    'rotate_270',        
    'flip_horizontal', 
    'flip_vertical',
    'flip_diagonal',
    'flip_antidiagonal'           
]

'''Key points from the Alphago paper
1. skip pass moves
2. augment 8 symmetries and reflections to the dataset
'''
class DataProcessor:
    def __init__(self, encoder, data_dir):
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_dir
    
    def process_sgf_files(self, zip_file_name = None, file_list = None):
        board_size = 19
        zip_file = None
        if zip_file_name is not None:
            # remove gzip compression
            tar_file = self.unzip_data(zip_file_name)
            # open tarfile
            zip_file = tarfile.open(self.data_dir + '/' + tar_file)
            # get name of all files
            file_list = zip_file.getnames()
        # total number of moves
        print(f"zip_file: {zip_file_name}")
        print(f"no of files: {len(file_list[1:])}")
        total_examples = self.num_total_examples(zip_file, file_list[1:])
        print(f"total examples: {total_examples}")
        shape = self.encoder.shape()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        print("feature shape: ")
        print(feature_shape)
        features = []
        labels = []
        
        
        for name in tqdm.tqdm(file_list[1:]):
            # read sgf content as string
            if zip_file:
                with zip_file.extractfile(name) as file:
                    sgf_content = file.read()
            else:
                with open(name, 'r') as file:
                    sgf_content = file.read()
            # Now we need to parse the string into a readable python object
            # create sgf game from string. 
            # parses the string and creates a Sgf_Game object
            sgf = Sgf_game.from_string(sgf_content)
            move = None
            # Now we get the sgf game object to replay each move
            # we play handicap moves first and get the game state
            game_state, first_move_done = self.get_handicap(sgf)
            
            # then we play the moves from the sequence
            # for every game state, we encode the game state and append to features
            # and next move we encode as label
            start = time.time()
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        # point has 1 based indexing
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                        # skipping first move
                        if first_move_done:
                            # augmenting 8 symmetrical transformations to features and labels - [2]
                            for transformation in transformations:
                                transformed_game_state = self.apply_transformation(game_state, transformation)
                                transformed_point = self.transform_point(point, transformation, 19)
                                features.append(self.encoder.encode(transformed_game_state)) 
                                labels.append(self.encoder.encode_point(transformed_point))
                    # skip on pass moves - [1]
                    else:
                        move = Move.pass_turn()    
                    game_state = game_state.apply_move(move)
                    first_move_done = True
            end = time.time()

        # Save the processed data
        base_name = zip_file_name.replace('.tar.gz', '') if zip_file_name else 'kgs-server-'
        data_file_name = self.data_dir + '/' + base_name
        train_feature_file_template = data_file_name + '_train_features'
        train_label_file_template = data_file_name + '_train_labels'
        test_feature_file_template = data_file_name + '_test_features'
        test_label_file_template = data_file_name + '_test_labels'

        indices = np.arange(len(features))
        # training to test split ratio 4:1, Alphago originally uses first 1 million
        # for the test while the rest 28.4 million for training. For simplicity we just split
        # our set to 4:1 ration instead of 28:1
        split_idx = int(0.8 * len(features))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        features = np.stack(features, axis=0)
        labels = np.asarray(labels, dtype=np.int16)

        X_train = features[train_indices]
        X_test = features[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]

        np.save(train_feature_file_template, X_train)
        np.save(train_label_file_template, y_train)
        np.save(test_feature_file_template, X_test)
        np.save(test_label_file_template, y_test)

        return
        
    """Total number of moves each player plays throughout all
       the games provided.
    """
    def num_total_examples(self, zip_file, file_list):
        total_examples = 0
        for name in file_list:
            # read the file if the file is an sgf file
            if name.endswith('.sgf'):
                if zip_file:
                    with zip_file.extractfile(name) as file:
                        sgf_content = file.read()
                else:
                    with open(name, 'r') as file:
                        sgf_content = file.read()
                # get an sgf string from the file for further processing
                sgf = Sgf_game.from_string(sgf_content)

                # this is important because we need to start counting after the first move
                # if the game has handicap, then main sequence starts with 2nd move. otherwise,
                # main sequence starts with first move and we need to skip it
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
        return total_examples
    
    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(self.data_dir + '/' + zip_file_name)  # <1>

        tar_file = zip_file_name[0:-3]  # <2>
        this_tar = open(self.data_dir + '/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)  # <3>
        this_tar.close()
        return tar_file
    
    @staticmethod
    def get_handicap(sgf):
        board_size = sgf.get_size()
        go_board = Board(board_size, board_size)
        first_move_done = False
        game_state = GameState.new_game(board_size)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for stone_tuple in setup:
                    row, col = stone_tuple
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            # Handicap stones aren't moves, so last_move should be None
            game_state = GameState(go_board, Player.white, None, None)
        return game_state, first_move_done
    
    def combine_numpy_files(self, base_name, file_type, output_filename=None, matching_files = None):
        """
        Generic function to combine multiple numpy files into a single file.
        
        Args:
            data_dir (str): Directory containing the numpy files
            base_name (str): Base name to match files (e.g., 'KGS-2019_04-19-1255-_train')
            file_type (str): Type of files to combine ('features' or 'labels')
            output_filename (str, optional): Output filename. If None, auto-generates one.
        
        Returns:
            str: Path to the combined output file
        """
        
        # Sort files to ensure consistent ordering
        matching_files.sort()
        print(f"Found {len(matching_files)} {file_type} files to combine:")
        # Load and combine all files
        combined_data = []
        for file in matching_files:
            file_path = os.path.join(self.data_dir, file)
            data = np.load(file_path)
            combined_data.append(data)
            print(f"  Loaded {file}: shape {data.shape}")
        
        # Concatenate along the first axis (assuming these are training samples)
        final_data = np.concatenate(combined_data, axis=0)
        print(f"Combined shape: {final_data.shape}")
        
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = f"{base_name}_{file_type}_combined.npy"
        
        output_path = os.path.join(self.data_dir, output_filename)
        
        # Save combined data
        np.save(output_path, final_data)
        print(f"Combined {file_type} saved to: {output_filename}")
        
        return output_path 
    def transform_point(self, point, transformation, board_size):
        """Transform a point according to the given transformation."""
        r, c = point.row, point.col
        N = board_size
        
        if transformation == 'identity':
            return Point(r, c)
        elif transformation == 'rotate_90':
            return Point(c, N + 1 - r)
        elif transformation == 'rotate_180':
            return Point(N + 1 - r, N + 1 - c)
        elif transformation == 'rotate_270':
            return Point(N + 1 - c, r)
        elif transformation == 'flip_horizontal':
            return Point(r, N + 1 - c)
        elif transformation == 'flip_vertical':
            return Point(N + 1 - r, c)
        elif transformation == 'flip_diagonal':
            return Point(c, r)
        elif transformation == 'flip_antidiagonal':
            return Point(N + 1 - c, N + 1 - r)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")  

    def apply_transformation(self, game_state, transformation):
        """Apply a D8 transformation to a game_state and its previous state chain."""
        board_size = game_state.board.num_rows
        new_board = Board(board_size, board_size)
        last_move = game_state.last_move
        if last_move is not None and last_move.is_play:
            last_point = last_move.point
            transformed_last_point = self.transform_point(last_point, transformation, board_size)
            last_move = Move.play(transformed_last_point)
        elif last_move is not None and last_move.is_pass:
            last_move = Move.pass_turn()
        elif last_move is not None and last_move.is_resign:
            last_move = Move.resign()
        # If last_move is None, keep it as None
        
        # Iterate through all points on the original board
        for row in range(1, board_size + 1):
            for col in range(1, board_size + 1):
                original_point = Point(row, col)
                stone_color = game_state.board.get(original_point)
                
                if stone_color is not None:
                    # Transform the point
                    transformed_point = self.transform_point(original_point, transformation, board_size)
                    # Place stone at transformed location
                    new_board.place_stone(stone_color, transformed_point)
        
        # Recursively transform the previous state if it exists
        transformed_previous_state = None
        # if game_state.previous_state is not None:
        #     transformed_previous_state = self.apply_transformation(
        #         game_state.previous_state, transformation
        #     )
        
        # Create new GameState with transformed board and previous state
        return GameState(new_board, game_state.next_player, transformed_previous_state, last_move)    

