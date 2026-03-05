from pydantic import BaseModel, Field
from typing import Optional


class ProcessRequest(BaseModel):
    file_id : str = None
    chunk_size : Optional[int] = Field(None, description="The size of each chunk in bytes.")
    overlap_size : Optional[int] = Field(None, description="The size of the overlap between chunks in bytes.")
    do_reset : Optional[int] = Field(0, description="Flag to indicate whether to reset the file processing state.")
