from pathlib import Path
from typing import Dict, List, Tuple

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer

from transformer_base import run_summarization
from utils import RelationData, RelationSentence


class Encoder(BaseModel):
    def encode_x(self, x: str) -> str:
        raise NotImplementedError

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        raise NotImplementedError

    def decode(self, x: str, y: str) -> RelationSentence:
        raise NotImplementedError

    def decode_x(self, x: str) -> str:
        raise NotImplementedError

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def encode_to_line(self, sent: RelationSentence) -> str:
        raise NotImplementedError

    def decode_from_line(self, line: str) -> RelationSentence:
        raise NotImplementedError

    def parse_line(self, line: str) -> Tuple[str, str]:
        raise NotImplementedError


class GenerateEncoder(Encoder):
    def encode_x(self, r: str) -> str:
        return f"Relation : {r} ."

    def decode_x(self, text: str) -> str:
        return text.split("Relation : ")[-1][:-2]

    def encode_triplet(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return "Context : " + sent.text + " Head Entity : " + s + " , Tail Entity : " + o + " ."

    def decode_triplet(self, text: str, label: str) -> RelationSentence:
        print("text",text)
        try:
            front, back = text.split(" Head Entity : ")
            _, context = front.split("Context : ")
            print("back",back)
        except ValueError:
            back = "nothing"
            context = "error"
        try:     
            head, back = back.split(", Tail Entity : ") 
        except ValueError:     
            head = back     
            back = "nothing"

        tail = back[:-2]
        return RelationSentence.from_spans(context, head, tail, label)

    def encode_y(self, sent: RelationSentence) -> str:
        return self.encode_x(sent.label) + " " + self.encode_triplet(sent)

    def decode_y(self, text: str, label: str) -> RelationSentence:
        del label
        try:
            front, back = text.split(" . Context : ")
        except ValueError:
            front = "not right"
            back = "mistakes"
        label = self.decode_x(front + " .")
        return self.decode_triplet("Context : " + back, label)

            

    def decode(self, x: str, y: str) -> RelationSentence:
        r = self.decode_x(x)
        sent = self.decode_y(y, r)
        return sent

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.label)
        y = self.encode_y(sent)
        return x, y

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return y + "\n"

    def parse_line(self, line: str) -> Tuple[str, str]:
        return "", line.strip()


class ExtractEncoder(Encoder):
    def encode_x(self, text: str) -> str:
        return f"Context : {text}"

    def decode_x(self, x: str) -> str:
        return x.split("Context : ")[-1]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Head Entity : {s} , Tail Entity : {o} , Relation : {r} ."

    def decode_y(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(" , Relation : ")
        label = label[:-2]
        front, tail = front.split(" , Tail Entity : ")
        _, head = front.split("Head Entity : ")
        return RelationSentence.from_spans(context, head, tail, label)

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"Head Entity : {head} , Tail Entity : {tail} , Relation :"

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def decode(self, x: str, y: str) -> RelationSentence:
        return self.decode_y(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return run_summarization.encode_to_line(x, y)

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return run_summarization.decode_from_line(line)


def select_encoder(name: str) -> Encoder:
    mapping: Dict[str, Encoder] = dict(
        extract=ExtractEncoder(),
        generate=GenerateEncoder(),
    )
    encoder = mapping[name]
    return encoder



if __name__ == "__main__":
    Fire()
