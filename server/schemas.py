from pydantic import BaseModel
from typing import List


class SceneObject(BaseModel):
    path: str
    class_name: str
    id: int


class SceneEdge(BaseModel):
    source: str
    source_name: str
    target: str
    target_name: str
    relation: str


class SceneGraph(BaseModel):
    objects: List[SceneObject]
    edges: List[SceneEdge]
