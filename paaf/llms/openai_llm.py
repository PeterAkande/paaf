import os
from dotenv import load_dotenv
import openai

from paaf.llms.base_llm import BaseLLM


load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY", None)


class OpenAILLM(BaseLLM):
    """
    OpenAI Language Model Wrapper
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str = None,
        api_key: str = None,
    ):
        super().__init__()

        self.api_key = api_key or OPEN_AI_API_KEY
        self.base_url = base_url
        self.model = model
        self.client = openai.Client(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, response_format=None) -> str:
        """
        Generate a response based on the provided prompt.

        Args:
            prompt (str): The prompt to generate a response for.
            response_format: The format of the response, if any.

        Returns:
            str: The generated response.
        """

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format if response_format else openai.NOT_GIVEN,
            max_tokens=1000,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
