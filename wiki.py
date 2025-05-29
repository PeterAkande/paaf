import wikipediaapi


class WikiPediaAPIClient:
    """
    A client for interacting with the Wikipedia API.
    """

    def __init__(self, language: str = "en"):
        self.wiki_wiki = wikipediaapi.Wikipedia(
            user_agent="ReAct Agent(akandepeter86@gmail.com)",
            language=language,
        )

    def __call__(self, *args, **kwds):
        """
        Allows the instance to be called like a function.
        """
        if len(args) == 1 and isinstance(args[0], str):
            # Get the first argument as the search query
            return self.search(args[0], **kwds)
        else:
            raise ValueError("Invalid arguments. Expected a single string query.")

    def search(self, query: str) -> str:
        """
        Perform a search using the Wikipedia API.

        Args:
            query (str): The search query.

        Returns:
            str: The summary of the search results.
        """
        page = self.wiki_wiki.page(query)

        if not page.exists():
            return None, "No results found."

        result = {
            "title": page.title,
            "summary": page.summary,
            "url": page.fullurl,
            "query": query,
        }
        return result, "Search successful."
    

def wiki_search(query: str, language: str = "en") -> dict:
    """
    Perform a search using the Wikipedia API.

    Args:
        query (str): The search query.
        language (str): The language of the Wikipedia page.

    Returns:
        dict: A dictionary containing the title, summary, and URL of the search result.
    """
    wiki_client = WikiPediaAPIClient(language=language)
    result, message = wiki_client(query)
    
    if result:
        return result
    else:
        raise ValueError(message)


if __name__ == "__main__":
    queries = ["Python (programming language)", "JavaScript", "Machine Learning"]
    wiki_client = WikiPediaAPIClient(language="en")
    for query in queries:
        result, message = wiki_client(query)
        if result:
            print(f"Title: {result['title']}")
            print(
                f"Summary: {result['summary'][:200]}..."
            )  # Print first 200 characters
            print(f"URL: {result['url']}")
        else:
            print(message)
        print()
