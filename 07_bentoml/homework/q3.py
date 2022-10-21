from pydantic import BaseModel, StrictStr 

class UserProfile(BaseModel):
    name: StrictStr
    age: int 
    country: StrictStr
    rating: float