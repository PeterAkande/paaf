# Serper handles the interaction with the Serper API to fetch search results.
import json
import os
from typing import Any, Dict, List
from paaf.config.logging import get_logger


import requests

from dotenv import load_dotenv


logger = get_logger(__name__)

load_dotenv()


SERPER_API_KEY = os.getenv("SERPER_API_KEY")


class SerpAPIClient:
    """
    A client for interacting with the Serper API to fetch google search results.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"

    def __call__(self, *args, **kwds):
        """
        Allows the instance to be called like a function.
        """
        if len(args) == 1 and isinstance(args[0], str):
            # Get the first argument as the search query
            return self.search(args[0], **kwds)
        else:
            raise ValueError("Invalid arguments. Expected a single string query.")

    def search(self, query: str, num_results: int = 10) -> dict:
        """
        Perform a search using the Serper API.

        Args:
            query (str): The search query.
            num_results (int): The number of results to return.

        Returns:
            dict: The search results.
        """
        payload = json.dumps(
            {
                "q": query,
                "num": num_results,
            }
        )

        header = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        response = requests.post(self.base_url, data=payload, headers=header)
        response.raise_for_status()
        return response.json()


def format_top_search_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format the top search results from the Serper API response.

    Args:
        results (Dict[str, Any]): The raw search results from the Serper API.

    Returns:
        List[Dict[str, Any]]: A list of formatted search results.
    """
    if "organic" not in results:
        logger.warning("No organic search results found.")
        return []

    formatted_results = []
    for item in results["organic"]:
        formatted_results.append(
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position", 0),
            }
        )

    return formatted_results


def search(q: str, location: str = "") -> List[Dict[str, Any]]:
    """
    Perform a search using the Serper API and format the results.

    Args:
        q (str): The search query.
        location (str): The location for the search (optional).

    Returns:
        List[Dict[str, Any]]: A list of formatted search results.
    """
    client = SerpAPIClient(SERPER_API_KEY)

    if location:
        q += f" location:{location}"

    results = client(q, num_results=10)

    formatted_results = format_top_search_results(results)
    return formatted_results


if __name__ == "__main__":
    # Example usage
    query = "OpenAI GPT-3"
    location = "San Francisco"
    results = search(query, location)

    for result in results:
        print(f"Title: {result['title']}")
        print(f"Link: {result['link']}")
        print(f"Snippet: {result['snippet']}")
        print(f"Position: {result['position']}\n")
