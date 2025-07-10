from abc import ABC, abstractmethod
import copy
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

from xllmx.model.tokenizer import Tokenizer
import xllmx.util.misc as misc

logger = logging.getLogger(__name__)


class LabelAllZeroError(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"LabelAllZeroError: {self.message}"


class ItemProcessorBase(ABC):
    @abstractmethod
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        raise NotImplementedError

    def predict_item_token_length(self, data_item: dict) -> int:
        """
        estimate the token length of the data item for gathering items of similar lengths into a batch
        """
        return 1


class MMConvItemProcessor(ItemProcessorBase):
    def __init__(
        self,
        transform: Dict[str, Callable[[Any], Dict]],
        media_symbols: List[str],   # like ["<|image|>", "<|video|>"],
        tokenizer: str | Tokenizer,  # 
        conv_template,
    ):
        self.transform = transform
        logger.info(f"transform:\n{self.transform}")

        self.media_symbols = media_symbols
        logger.info(f"media_symbols:\n{self.media_symbols}")

        if isinstance(tokenizer, str):
            self.tokenizer = Tokenizer(model_path=tokenizer)
        else:
            self.tokenizer = copy.deepcopy(tokenizer)

        # todo should not already exist
        self.tokenizer.tokenizer.add_tokens(media_symbols)
        self.d_media_symbol2token = {}  # {'<|image|>': 65536, '<|video|>': 65537}
        self.d_media_token2symbol = {}  # {65536: '<|image|>', 65537: '<|video|>'}
        for media_symbol in media_symbols:
            tokenized_symbol = self.tokenizer.encode(media_symbol, bos=False, eos=False)
            assert len(tokenized_symbol) == 1
            self.d_media_symbol2token[media_symbol] = tokenized_symbol[0]
            self.d_media_token2symbol[tokenized_symbol[0]] = media_symbol
        logger.info(f"d_media_symbol2token {self.d_media_symbol2token},  d_media_token2symbol {self.d_media_token2symbol}")

        # implicit_at_beginning means media without explict location specification are arranged right after bos token
        # if false, then these medias are arranged at the beginning of the first question
        self.implicit_at_beginning = False
        self.conv_template = conv_template

        # Add video-related tokens
        if "<|video|>" in self.media_symbols or "<|partial_video|>" in self.media_symbols:
            self.tokenizer.tokenizer.add_tokens(["<video_resolution_placeholder>", "<video_frames_placeholder>", "<video_fps_placeholder>"])
            # I do not use the format of "<|...|>" because I do not want the system to recognize me these palce holder as media tokens.
        

    def collect_and_process_media(self, data_item):
        """
        this function receives a raw piece of data (e.g. read from `.json` data file),
        and returns d_media, containing the prepared media (after transform) readily usable by model

        YOU MAY OVERRIDE THIS FUNCTION TO SUPPORT COMPLEX LOADING OF VARIOUS FORMS OF DATA
        
        returns a dict like
        {'<|image|>': [{'input_ids': result_toks, 'labels': result_toks, 'type': '<|image|>'}, {image2}]}
        """
        # import ipdb
        # ipdb.set_trace()

        d_media = {}
        for media_symbol in self.media_symbols:
            if media_symbol in data_item:
                l_media = data_item[media_symbol]  # a list of media paths
            elif media_symbol.lstrip("<|").rstrip("|>") in data_item:
                l_media = data_item[media_symbol.lstrip("<|").rstrip("|>")]
            else:
                l_media = []
            
            if not isinstance(l_media, list):  # data with only one media, in format {"image": image_name, ...}
                l_media = [l_media]

            d_media[media_symbol] = []
            for media in l_media:
                media = self.transform[media_symbol](media)
                assert isinstance(media, Dict)
                media["type"] = media_symbol
                d_media[media_symbol].append(media)

        return d_media

    def replace_media_token_with_media(
        self, tokens: List[int], labels: Union[List[int], None], d_media: Dict[str, List]
    ):
        d_media_counter = {key: 0 for key in d_media}

        # print(f"d_media_symbol2token {d_media_symbol2token},  d_media_token2symbol {d_media_token2symbol}")
        # self.d_media_symbol2token = {}  # {'<|image|>': 65536, '<|video|>': 65537}
        # self.d_media_token2symbol = {}. # {65536: '<|image|>', 65537: '<|video|>'}
        for i, t in enumerate(tokens):
            if t in self.d_media_token2symbol:
                media_symbol = self.d_media_token2symbol[t]
                media = d_media[media_symbol][d_media_counter[media_symbol]]
                d_media_counter[media_symbol] += 1
                tokens[i] = media
                media["to_predict"] = labels[i] > 0

        assert all([d_media_counter[key] == len(d_media[key]) for key in d_media])

        if labels is not None:
            return tokens, labels
        else:
            return tokens

    @staticmethod
    def insert_implicit_media_symbol_in_q1(conv_list: List[Dict], d_media: Dict):
        """
        Add the media tokens to the beginning of the first instruction from
        human. This logic may be more reasonable. However, it is incompatible
        with old-version Accessory models, which are trained with image tokens
        inserted directly behind the first token (<bos>).
        :param conv_list: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
        :param d_media: a dict of media for all media types
        """
        conv_list = copy.deepcopy(conv_list)

        for media_symbol, l_media in d_media.items():
            media_symbol_count = "".join([_["value"] for _ in conv_list if _["value"] is not None]).count(media_symbol)
            if media_symbol_count > 0:
                assert media_symbol_count == len(
                    l_media
                ), f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
            else:
                conv_list[0]["value"] = (media_symbol + " ") * len(l_media) + conv_list[0]["value"]

        return conv_list

    @staticmethod
    def insert_implicit_media_symbol_at_beginning(conv: str, d_media: Dict):
        """
        Legacy versions of LLaMA2-Accessory handled media in a non-interleaved
        manner, where image tokens are inserted directly behind the first token,
        namely <bos>. To support interleaved media comprehension and generation,
        Accessory now supports the explicit specification of media occurrence,
        which is achieved by adding media symbols, e.g. <image>, within the
        conversations. On the other hand, for media without explicit
        specification, this function realizes the legacy behavior to arrange
        them at the beginning of the conversation.
        :param conv: conversation
        :param d_media: a dict of media for all media types, for determining how
        many media tokens need to be inserted
        """
        conv = copy.deepcopy(conv)

        for media_symbol, l_media in d_media.items():
            media_symbol_count = conv.count(media_symbol)
            if media_symbol_count > 0:
                assert media_symbol_count == len(
                    l_media
                ), f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
            else:
                conv = (media_symbol + " ") * len(l_media) + conv

        return conv

    def preprocess_item(self, data_item):
        return data_item

    def add_speaker_and_signal(self, source: List):
        """
        Given source instruction and response pieces, return the text containing the complete conversation,
        and the list of values that the model should learn to predict during training
        :param source: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
        :return: `conversation`: string containing the complete conversation;
                 `to_predict_list`: the list of values that the model should learn to predict during training
        """
        conv = self.conv_template()

        for i, sentence in enumerate(source):
            from_str = sentence["from"]
            if i % 2 == 0:
                assert from_str.lower() in ["human"]
                role = conv.roles[0]
            elif i % 2 == 1:
                assert from_str.lower() in ["gpt", "assistant"]
                role = conv.roles[1]
            else:
                raise ValueError(f"unknown dialog role: {from_str.lower()}")

            value = sentence["value"]

            conv.append_message(role, value)

        processed = conv.process()
        conversation, pieces = processed["conv"], processed["pieces"]

        return conversation, pieces

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        data_item = self.preprocess_item(data_item)

        d_media = self.collect_and_process_media(data_item)

        source = data_item["conversations"]
        
        ### Save a dict to a local file. {"1": [token], "2": [token], ......, "999": [token]}
        # import ipdb
        # ipdb.set_trace()
        # token_dict = {}
        # for i in range(0,1000):
        #     token_dict[str(i)] = self.tokenizer.encode(str(i), bos=False, eos=False)
        # with open('/mnt/workspace/workgroup/hangjie.yhj/code/Lumina-mGPT-main/lumina_mgpt/data/num2tokens.json', 'w') as f:
        #     json.dump(token_dict, f)

        ### replace the <video_resolution_placeholder> in conversation with explicit shape
        # New, support '<|partial_video|>' and '<|video|>'
        if '<|video|>' in d_media:
            if len(d_media['<|video|>']) > 0:
                source = self.replace_resolution_token(d_media, source, '<|video|>')
        elif '<|partial_video|>' in d_media:
            if len(d_media['<|partial_video|>']) > 0:
                source = self.replace_resolution_token(d_media, source, '<|partial_video|>')
        print(f"source {source}\n")

        # Original, only support '<|video|>'
        # if len(d_media['<|video|>']) > 0:
        #     source = self.replace_resolution_token(d_media, source)
        # print(f"source {source}")


        # implicit_at_beginning means media without explict location specification are arranged right after bos token
        # if false, then these medias are arranged at the beginning of the first question
        if not self.implicit_at_beginning:
            # Adds media symbols to the start of the first "human" message if they are not already explicitly included (usually it should be included).
            source = self.insert_implicit_media_symbol_in_q1(source, d_media)

        conversation, pieces = self.add_speaker_and_signal(source)
        # add sep_token ("<reserved08706>") to conversation (add between conversations and at the end)
        # conversation: The full string of the conversation, 
        #        like "Generate an image of 768x768 according to the following prompt:\n This image is a promotional poster for Taylor Swift's 'The Eras Tour'. It features a stylized illustration of a blonde woman in a sparkling blue and purple sequined bodysuit with a high-cut leg. The figure is posed dramatically against a swirling pink and purple background. Her hair is long and straight, and she has bright red lips in an open-mouthed expression. The artwork is signed 'Anna W.' in the top right corner. At the bottom of the image, 'TAYLOR SWIFT THE ERAS TOUR' is written in large, bold black text. The overall aesthetic is glamorous and eye-catching, typical of concert tour promotional material.<reserved08706><|image|><reserved08706>"
        # pieces: pieces: A list of smaller chunks (e.g., individual turns) in the conversation, each with its corresponding role (human or assistant) and the part that needs to be predicted, 
        #        like[{'data': "Generate an image of 768x768 according to the following prompt:\n This image is a promotional poster for Taylor Swift's 'The Eras Tour'. It features a stylized illustration of a blonde woman in a sparkling blue and purple sequined bodysuit with a high-cut leg. The figure is posed dramatically against a swirling pink and purple background. Her hair is long and straight, and she has bright red lips in an open-mouthed expression. The artwork is signed 'Anna W.' in the top right corner. At the bottom of the image, 'TAYLOR SWIFT THE ERAS TOUR' is written in large, bold black text. The overall aesthetic is glamorous and eye-catching, typical of concert tour promotional material.<reserved08706>", 'predict': False}, {'data': '<|image|><reserved08706>', 'predict': True}]


        if self.implicit_at_beginning:
            conversation = self.insert_implicit_media_symbol_at_beginning(conversation, d_media)

        # dialog does not need eos
        tokens = self.tokenizer.encode(conversation, bos=True, eos=False) # bos: begin of sentence; eos: end of sentence
        labels = [-100 for _ in tokens]

        # check special token num as expected
        for media_symbol, l_media in d_media.items():
            media_token = self.d_media_symbol2token[media_symbol]
            media_token_count = tokens.count(media_token)
            assert media_token_count == len(l_media), (
                f"{media_token_count} {media_token} (for {media_symbol}) exists in tokenized conversation, "
                f"but {len(l_media)} actual media are given"
            )

        # This step verifies that the tokenized version of each "piece" matches the corresponding part of the full tokenized conversation. If a piece is marked as needing to be predicted (p["predict"]), the labels are updated to reflect the correct tokens for the model to predict.
        check_pos = 0
        for i, p in enumerate(pieces):
            if i == 0:
                tokenized_value = self.tokenizer.encode(p["data"], bos=True, eos=False)
            else:
                tokenized_value = self.tokenizer.encode_wo_prefix_space(p["data"])

            assert (
                tokens[check_pos : check_pos + len(tokenized_value)] == tokenized_value
            ), "inconsistent complete conversation and corresponding piece after tokenization"

            if p["predict"]:
                labels[check_pos : check_pos + len(tokenized_value)] = tokenized_value

            check_pos = check_pos + len(tokenized_value)

        if training_mode and all([_ <= 0 for _ in labels]):  # nothing to predict
            raise LabelAllZeroError()

        # labels will be processed later by the model
        tokens, labels = self.replace_media_token_with_media(tokens, labels, d_media)

        assert len(tokens) == len(labels)

        if training_mode:
            return tokens, labels
        else:
            return tokens

    def predict_item_token_length(self, data_item: dict) -> int:
        """
        estimate the length of each item
        """

        if "conversations" in data_item:
            return sum([len(_["value"]) for _ in data_item["conversations"]])
        else:
            return 1
    

    # def replace_resolution_token(self, d_media, source) -> dict:
        
    #     conver_idx = 0
    #     for m in d_media['<|video|>']:
    #         # we iteratively do the replacement for 'conversation', so that we just need to set the 'n' parameter to 1.
    #         # n: 1-based index
    #         while conver_idx<=len(source):
    #             if source[conver_idx]['from']=='human':
    #                 source[conver_idx]['value'] = misc.replace_nth_str(
    #                                     string = source[conver_idx]['value'],
    #                                     sub = "<video_resolution_placeholder>",
    #                                     wanted = f"{int(m['size_wh'][0])}x{int(m['size_wh'][1])}",
    #                                     n = 1) 
    #                 conver_idx += 1
    #                 break
    #     return source
    
    def replace_resolution_token(self, d_media: dict, source: list, video_key: str) -> list:
        """
        Replaces the <video_resolution_placeholder> in 'source' with the actual resolution 
        from the 'd_media' dictionary for each video. The replacement occurs iteratively for 
        'human' conversation entries.

        Parameters:
        d_media (dict): A dictionary containing video metadata, where each video entry has the key '<|video|>'.
                        Each video contains 'size_wh', a tuple with width and height.
        source (list): A list of dictionaries, each representing a conversation entry. The entries contain
                    a 'from' key indicating the source ('human' or other) and a 'value' key with the conversation content.
        video_key (str): '<|video|>' or '<|partial_video|>'

        Returns:
        list: The modified source list with the <video_resolution_placeholder> replaced by the actual resolution (e.g., '1920x1080')
            for conversation entries from 'human'.
        """
        
        # video_key (str): '<|video|>' or '<|partial_video|>'

        conver_idx = 0  # Initialize the conversation index
        for video in d_media[video_key]:  # Iterate over each video in the media data
            # Continue searching through the 'source' list until we find a 'human' entry
            while conver_idx < len(source):
                if source[conver_idx]['from'] == 'human':
                    # Replace the first occurrence of the resolution placeholder
                    # we iteratively do the replacement for 'conversation', so that we just need to set the 'n' parameter to 1.
                    # n: 1-based index
                    resolution = f"{int(video['size_wh'][0])}x{int(video['size_wh'][1])}"
                    source[conver_idx]['value'] = misc.replace_nth_str(
                        string=source[conver_idx]['value'],
                        sub="<video_resolution_placeholder>",
                        wanted=resolution,
                        n=1
                    )
                    conver_idx += 1  # Move to the next conversation entry
                    break  # Proceed to the next video after one replacement
        return source





# class MMConvItemProcessor2(ItemProcessorBase):
#     def __init__(
#         self,
#         transform: Dict[str, Callable[[Any], Dict]],
#         media_symbols: List[str],   # like ["<|image|>", "<|video|>"],
#         tokenizer: str | Tokenizer,  # 
#         conv_template,
#     ):
#         self.transform = transform
#         logger.info(f"transform:\n{self.transform}")

#         self.media_symbols = media_symbols
#         logger.info(f"media_symbols:\n{self.media_symbols}")

#         if isinstance(tokenizer, str):
#             self.tokenizer = Tokenizer(model_path=tokenizer)
#         else:
#             self.tokenizer = copy.deepcopy(tokenizer)

#         # todo should not already exist
#         self.tokenizer.tokenizer.add_tokens(media_symbols)
#         self.d_media_symbol2token = {}  # {'<|image|>': 65536, '<|video|>': 65537}
#         self.d_media_token2symbol = {}  # {65536: '<|image|>', 65537: '<|video|>'}
#         for media_symbol in media_symbols:
#             tokenized_symbol = self.tokenizer.encode(media_symbol, bos=False, eos=False)
#             assert len(tokenized_symbol) == 1
#             self.d_media_symbol2token[media_symbol] = tokenized_symbol[0]
#             self.d_media_token2symbol[tokenized_symbol[0]] = media_symbol
#         logger.info(f"d_media_symbol2token {self.d_media_symbol2token},  d_media_token2symbol {self.d_media_token2symbol}")

#         # implicit_at_beginning means media without explict location specification are arranged right after bos token
#         # if false, then these medias are arranged at the beginning of the first question
#         self.implicit_at_beginning = False
#         self.conv_template = conv_template

#         # Add video-related tokens
#         if "<|video|>" in self.media_symbols:
#             self.tokenizer.tokenizer.add_tokens(["<video_resolution_placeholder>", "<video_frames_placeholder>", "<video_fps_placeholder>"])
#             # I do not use the format of "<|...|>" because I do not want the system to recognize me these palce holder as media tokens.
        

#     def collect_and_process_media(self, data_item):
#         """
#         this function receives a raw piece of data (e.g. read from `.json` data file),
#         and returns d_media, containing the prepared media (after transform) readily usable by model

#         YOU MAY OVERRIDE THIS FUNCTION TO SUPPORT COMPLEX LOADING OF VARIOUS FORMS OF DATA
        
#         returns a dict like
#         {'<|image|>': [{'input_ids': result_toks, 'labels': result_toks, 'type': '<|image|>'}, {image2}]}
#         """
#         d_media = {}
#         for media_symbol in self.media_symbols:
#             if media_symbol in data_item:
#                 l_media = data_item[media_symbol]  # a list of media paths
#             elif media_symbol.lstrip("<|").rstrip("|>") in data_item:
#                 l_media = data_item[media_symbol.lstrip("<|").rstrip("|>")]
#             else:
#                 l_media = []
#             if not isinstance(l_media, list):  # data with only one media, in format {"image": image_name, ...}
#                 l_media = [l_media]

#             d_media[media_symbol] = []
#             for media in l_media:
#                 media = self.transform[media_symbol](media)
#                 assert isinstance(media, Dict)
#                 media["type"] = media_symbol
#                 d_media[media_symbol].append(media)

#         return d_media

#     def replace_media_token_with_media(
#         self, tokens: List[int], labels: Union[List[int], None], d_media: Dict[str, List]
#     ):
#         d_media_counter = {key: 0 for key in d_media}

#         # print(f"d_media_symbol2token {d_media_symbol2token},  d_media_token2symbol {d_media_token2symbol}")
#         # self.d_media_symbol2token = {}  # {'<|image|>': 65536, '<|video|>': 65537}
#         # self.d_media_token2symbol = {}. # {65536: '<|image|>', 65537: '<|video|>'}
#         for i, t in enumerate(tokens):
#             if t in self.d_media_token2symbol:
#                 media_symbol = self.d_media_token2symbol[t]
#                 media = d_media[media_symbol][d_media_counter[media_symbol]]
#                 d_media_counter[media_symbol] += 1
#                 tokens[i] = media
#                 media["to_predict"] = labels[i] > 0

#         assert all([d_media_counter[key] == len(d_media[key]) for key in d_media])

#         if labels is not None:
#             return tokens, labels
#         else:
#             return tokens

#     @staticmethod
#     def insert_implicit_media_symbol_in_q1(conv_list: List[Dict], d_media: Dict):
#         """
#         Add the media tokens to the beginning of the first instruction from
#         human. This logic may be more reasonable. However, it is incompatible
#         with old-version Accessory models, which are trained with image tokens
#         inserted directly behind the first token (<bos>).
#         :param conv_list: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
#         :param d_media: a dict of media for all media types
#         """
#         conv_list = copy.deepcopy(conv_list)

#         for media_symbol, l_media in d_media.items():
#             media_symbol_count = "".join([_["value"] for _ in conv_list if _["value"] is not None]).count(media_symbol)
#             if media_symbol_count > 0:
#                 assert media_symbol_count == len(
#                     l_media
#                 ), f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
#             else:
#                 conv_list[0]["value"] = (media_symbol + " ") * len(l_media) + conv_list[0]["value"]

#         return conv_list

#     @staticmethod
#     def insert_implicit_media_symbol_at_beginning(conv: str, d_media: Dict):
#         """
#         Legacy versions of LLaMA2-Accessory handled media in a non-interleaved
#         manner, where image tokens are inserted directly behind the first token,
#         namely <bos>. To support interleaved media comprehension and generation,
#         Accessory now supports the explicit specification of media occurrence,
#         which is achieved by adding media symbols, e.g. <image>, within the
#         conversations. On the other hand, for media without explicit
#         specification, this function realizes the legacy behavior to arrange
#         them at the beginning of the conversation.
#         :param conv: conversation
#         :param d_media: a dict of media for all media types, for determining how
#         many media tokens need to be inserted
#         """
#         conv = copy.deepcopy(conv)

#         for media_symbol, l_media in d_media.items():
#             media_symbol_count = conv.count(media_symbol)
#             if media_symbol_count > 0:
#                 assert media_symbol_count == len(
#                     l_media
#                 ), f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
#             else:
#                 conv = (media_symbol + " ") * len(l_media) + conv

#         return conv

#     def preprocess_item(self, data_item):
#         return data_item

#     def add_speaker_and_signal(self, source: List):
#         """
#         Given source instruction and response pieces, return the text containing the complete conversation,
#         and the list of values that the model should learn to predict during training
#         :param source: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
#         :return: `conversation`: string containing the complete conversation;
#                  `to_predict_list`: the list of values that the model should learn to predict during training
#         """
#         conv = self.conv_template()

#         for i, sentence in enumerate(source):
#             from_str = sentence["from"]
#             if i % 2 == 0:
#                 assert from_str.lower() in ["human"]
#                 role = conv.roles[0]
#             elif i % 2 == 1:
#                 assert from_str.lower() in ["gpt", "assistant"]
#                 role = conv.roles[1]
#             else:
#                 raise ValueError(f"unknown dialog role: {from_str.lower()}")

#             value = sentence["value"]

#             conv.append_message(role, value)

#         processed = conv.process()
#         conversation, pieces = processed["conv"], processed["pieces"]

#         return conversation, pieces

#     def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
#         data_item = self.preprocess_item(data_item)

#         # breakpoint()
#         # import pdb; pdb.set_trace()
#         # print('MMConvItemProcessor')
#         d_media = self.collect_and_process_media(data_item)

#         source = data_item["conversations"]

#         # implicit_at_beginning means media without explict location specification are arranged right after bos token
#         # if false, then these medias are arranged at the beginning of the first question
#         if not self.implicit_at_beginning:
#             # Adds media symbols to the start of the first "human" message if they are not already explicitly included (usually it should be included).
#             source = self.insert_implicit_media_symbol_in_q1(source, d_media)

#         conversation, pieces = self.add_speaker_and_signal(source)
#         # add sep_token ("<reserved08706>") to conversation (add between conversations and at the end)
#         # conversation: The full string of the conversation, 
#         #        like "Generate an image of 768x768 according to the following prompt:\n This image is a promotional poster for Taylor Swift's 'The Eras Tour'. It features a stylized illustration of a blonde woman in a sparkling blue and purple sequined bodysuit with a high-cut leg. The figure is posed dramatically against a swirling pink and purple background. Her hair is long and straight, and she has bright red lips in an open-mouthed expression. The artwork is signed 'Anna W.' in the top right corner. At the bottom of the image, 'TAYLOR SWIFT THE ERAS TOUR' is written in large, bold black text. The overall aesthetic is glamorous and eye-catching, typical of concert tour promotional material.<reserved08706><|image|><reserved08706>"
#         # pieces: pieces: A list of smaller chunks (e.g., individual turns) in the conversation, each with its corresponding role (human or assistant) and the part that needs to be predicted, 
#         #        like[{'data': "Generate an image of 768x768 according to the following prompt:\n This image is a promotional poster for Taylor Swift's 'The Eras Tour'. It features a stylized illustration of a blonde woman in a sparkling blue and purple sequined bodysuit with a high-cut leg. The figure is posed dramatically against a swirling pink and purple background. Her hair is long and straight, and she has bright red lips in an open-mouthed expression. The artwork is signed 'Anna W.' in the top right corner. At the bottom of the image, 'TAYLOR SWIFT THE ERAS TOUR' is written in large, bold black text. The overall aesthetic is glamorous and eye-catching, typical of concert tour promotional material.<reserved08706>", 'predict': False}, {'data': '<|image|><reserved08706>', 'predict': True}]

#         if self.implicit_at_beginning:
#             conversation = self.insert_implicit_media_symbol_at_beginning(conversation, d_media)

#         # dialog does not need eos
#         tokens = self.tokenizer.encode(conversation, bos=True, eos=False) # bos: begin of sentence; eos: end of sentence
#         labels = [-100 for _ in tokens]

#         # check special token num as expected
#         for media_symbol, l_media in d_media.items():
#             media_token = self.d_media_symbol2token[media_symbol]
#             media_token_count = tokens.count(media_token)
#             assert media_token_count == len(l_media), (
#                 f"{media_token_count} {media_token} (for {media_symbol}) exists in tokenized conversation, "
#                 f"but {len(l_media)} actual media are given"
#             )

#         # This step verifies that the tokenized version of each "piece" matches the corresponding part of the full tokenized conversation. If a piece is marked as needing to be predicted (p["predict"]), the labels are updated to reflect the correct tokens for the model to predict.
#         check_pos = 0
#         for i, p in enumerate(pieces):
#             if i == 0:
#                 tokenized_value = self.tokenizer.encode(p["data"], bos=True, eos=False)
#             else:
#                 tokenized_value = self.tokenizer.encode_wo_prefix_space(p["data"])

#             assert (
#                 tokens[check_pos : check_pos + len(tokenized_value)] == tokenized_value
#             ), "inconsistent complete conversation and corresponding piece after tokenization"

#             if p["predict"]:
#                 labels[check_pos : check_pos + len(tokenized_value)] = tokenized_value

#             check_pos = check_pos + len(tokenized_value)

#         if training_mode and all([_ <= 0 for _ in labels]):  # nothing to predict
#             raise LabelAllZeroError()

#         # labels will be processed later by the model
#         tokens, labels = self.replace_media_token_with_media(tokens, labels, d_media)

#         assert len(tokens) == len(labels)

#         if training_mode:
#             return tokens, labels
#         else:
#             return tokens

#     def predict_item_token_length(self, data_item: dict) -> int:
#         """
#         estimate the length of each item
#         """

#         if "conversations" in data_item:
#             return sum([len(_["value"]) for _ in data_item["conversations"]])
#         else:
#             return 1
