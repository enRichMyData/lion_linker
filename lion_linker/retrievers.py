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

    async def fetch_entities(self, mention, session):
        raise NotImplementedError

    async def fetch_multiple_entities(self, mentions: list[str]):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_entities(mention, session) for mention in mentions]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle failed requests (e.g., those returning exceptions)
            output = {}
            for mention, result in zip(mentions, results):
                if isinstance(result, Exception):
                    output[mention] = f"Error: {str(result)}"
                else:
                    output[mention] = result
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

    async def fetch_entities(self, mention, session):
        params = {
            "name": mention,
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


class WikidataClient(RetrieverClient):
    def __init__(
        self,
        *args,
        language: str = "en",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.language = language

    async def fetch_entities(self, mention: str, session: aiohttp.ClientSession):
        """
        Given a mention and a session, fetch candidate entities (with labels and descriptions)
        and for each candidate retrieve its types (ordered by type label). Includes retries.
        """
        # First, fetch candidates
        candidates = await self._fetch_candidates(mention, session)
        # Then, fetch types for each candidate concurrently
        tasks = [self._fetch_candidate_types(candidate, session) for candidate in candidates]
        candidates_with_types = await asyncio.gather(*tasks)
        return candidates_with_types

    async def _post_query_with_retries(self, query: str, session: aiohttp.ClientSession):
        headers = {
            "User-Agent": "sparqlwrapper 2.0.0 (rdflib.github.io/sparqlwrapper)",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        retries = self.max_retries
        delay = self.backoff_factor
        while retries > 0:
            try:
                async with session.post(
                    self.endpoint, data={"query": query, "format": "json"}, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                await asyncio.sleep(delay)
                delay *= 2

    async def _fetch_candidates(self, mention: str, session: aiohttp.ClientSession):
        candidate_query = f"""
            SELECT ?item ?itemLabel ?itemDescription
            WHERE {{
            SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:api "EntitySearch" .
                bd:serviceParam wikibase:endpoint "www.wikidata.org" .
                bd:serviceParam mwapi:search "{mention}" .
                bd:serviceParam mwapi:language "{self.language}" .
                ?item wikibase:apiOutputItem mwapi:item.
            }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
            }}
            LIMIT {self.num_candidates}
        """
        data = await self._post_query_with_retries(candidate_query, session)
        candidates = []
        for result in data["results"]["bindings"]:
            item_uri = result["item"]["value"]
            candidate_id = item_uri.split("/")[-1]  # Extract QID
            candidate_label = result.get("itemLabel", {}).get("value", "")
            candidate_desc = result.get("itemDescription", {}).get("value", "")
            candidates.append(
                {
                    "id": candidate_id,
                    "name": candidate_label,
                    "description": candidate_desc,
                    "types": [],  # to be filled in next step
                }
            )
        return candidates

    async def _fetch_candidate_types(self, candidate: dict, session: aiohttp.ClientSession):
        candidate_id = candidate["id"]
        prefixes = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        """
        types_query = (
            prefixes
            + f"""
                SELECT ?type ?typeLabel
                WHERE {{
                wd:{candidate_id} wdt:P31 ?type.
                ?type rdfs:label ?typeLabel.
                FILTER(LANG(?typeLabel) = "{self.language}")
                }}
                ORDER BY ?typeLabel
            """
        )
        try:
            type_data = await self._post_query_with_retries(types_query, session)
            types = []
            for t in type_data["results"]["bindings"]:
                type_uri = t["type"]["value"]
                type_id = type_uri.split("/")[-1]
                type_label = t["typeLabel"]["value"]
                types.append({"id": type_id, "name": type_label})
            candidate["types"] = types
        except Exception as e:
            print(f"Error retrieving types for candidate {candidate_id}: {e}")
            candidate["types"] = []
        return candidate
