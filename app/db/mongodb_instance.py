from pymongo import MongoClient
from dotenv import load_dotenv
import os
from pymongo.errors import ConnectionFailure

load_dotenv()

class MongoDBInstance:
    """
    The code snippet implements a singleton pattern for creating a MongoDB instance with a shared
    connection.

    :param cls: The `cls` parameter in the code snippet refers to the class itself. In this case, it is
    referring to the class `MongoDBInstance`. The `cls` parameter is commonly used in class methods to
    access class variables and methods
    :return: The `get_database` method returns the database instance that was initialized in the
    `_init_connection` method.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBInstance, cls).__new__(cls)
            cls._instance._init_connection()
        return cls._instance

    def _init_connection(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client['agentic_rag_database']

    def get_database(self):
        return self.db

    def health_check(self):
        """
        Checks the health of the MongoDB connection.
        :return: True if the connection is healthy, otherwise raises an exception.
        """
        try:
            # Attempt to ping the database
            self.client.admin.command("ping")
            return True
        except ConnectionFailure as e:
            raise ConnectionFailure("MongoDB is not connected") from e
        except Exception as e:
            raise Exception(f"Unexpected error occurred: {str(e)}")