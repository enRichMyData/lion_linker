import asyncio
import logging
import traceback

from jsonargparse import ArgumentParser
from jsonargparse._util import import_object

from lion_linker.lion_linker import LionLinker
from lion_linker.retrievers import RetrieverClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main():
    parser = ArgumentParser()
    parser.add_class_arguments(LionLinker, "lion", skip={"retriever"})
    parser.add_subclass_arguments(RetrieverClient, "retriever", default="RetrieverClient")
    args = parser.parse_args()

    # Create retriever instance based on the subclass selected via the "cls" key.
    retriever_args = args.retriever.pop("init_args")
    retriever_class = args.retriever.pop("class_path")
    retriever_class = import_object(retriever_class)
    retriever_instance = retriever_class(**retriever_args)

    # Create LionLinker instance using the separate retriever instance.
    lion = LionLinker(retriever=retriever_instance, **args.lion)

    # Run the processing
    try:
        await lion.run()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
