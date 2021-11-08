class Anime:

    def __init__(self, anime_id: int, show_name: str, genre: str, 
                show_type: str, average_rating: float, 
                synopsis: str, title_image: dict) -> None:
        self.anime_id = anime_id
        self.show_name = show_name
        self.genre = genre
        self.show_type = show_type
        self.average_rating = average_rating
        self.synopsis = synopsis
        self.title_image = title_image

        self.embeddings: dict = {}