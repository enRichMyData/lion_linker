import asyncio

import aiohttp


class RetrieverClient:
    def __init__(
        self,
        endpoint: str,
        token: str = None,
        parse_response_func=None,
        max_retries=3,
        backoff_factor=0.5,
        num_candidates=10,
    ):
        self.endpoint = endpoint
        self.token = token
        self.parse_response_func = parse_response_func
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.num_candidates = num_candidates

    async def fetch_entities(self, query, session):
        raise NotImplementedError

    async def fetch_multiple_entities(self, queries):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_entities(query, session) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle failed requests (e.g., those returning exceptions)
            output = {}
            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    output[query] = f"Error: {str(result)}"
                else:
                    output[query] = result
            return output


class LamapiClient(RetrieverClient):
    def __init__(
        self,
        endpoint: str,
        token: str = None,
        parse_response_func=None,
        max_retries=3,
        backoff_factor=0.5,
        num_candidates=10,
        kg="wikidata",
        cache: bool = False,
    ):
        super().__init__(
            endpoint=endpoint,
            token=token,
            parse_response_func=parse_response_func,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            num_candidates=num_candidates,
        )
        self.kg = kg
        self.cache = cache

    async def fetch_entities(self, query, session):
        params = {
            "name": query,
            "limit": self.num_candidates,
            "kg": self.kg,
            "cache": str(self.cache),
        }  # Added kg parameter to request params
        if self.token:
            params["token"] = self.token

        retries = 0
        while retries < self.max_retries:
            try:
                async with session.get(self.endpoint, params=params, ssl=False) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    if self.parse_response_func:
                        return self.parse_response_func(response_json)
                    return response_json
            except aiohttp.ClientResponseError as e:
                if e.status in {502, 503, 504}:  # Server errors
                    retries += 1
                    wait_time = self.backoff_factor * (2**retries)  # Exponential backoff
                    print(f"Error {e.status}, retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"ClientResponseError: {e}")
                    raise
            except aiohttp.ClientConnectionError as e:
                print(f"ConnectionError: {e}")
                raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise

        raise Exception(f"Failed to fetch after {self.max_retries} retries")
