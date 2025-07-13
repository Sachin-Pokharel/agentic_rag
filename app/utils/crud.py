import logging
from pymongo.errors import PyMongoError
from db.mongodb_instance import MongoDBInstance


class ConversationStore:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.db = MongoDBInstance().get_database()
        self.collection = self.get_collection()

    def get_collection(self):
        """Check if the collection exists"""
        try:
            if self.collection_name in self.db.list_collection_names():
                logging.info(f"Using existing collection: {self.collection_name}")
                return self.db[self.collection_name]
        except PyMongoError as e:
            logging.error(f"Error accessing collection: {e}")
            raise Exception(f"Error accessing collection: {e}")
        
        
    def store_metadata(self, metadata: dict):
        """
        Stores metadata in the conversation_interview collection.
        """
        try:
            result = self.collection.insert_one(metadata)
            logging.info(f"Metadata saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logging.error(f"Error saving metadata: {e}")
            raise Exception(f"Error saving metadata: {e}")
        
        
    def save_booking(self, booking_data: dict):
        """
        Stores booking information in the booking_interview collection.
        """
        try:
            result = self.collection.insert_one(booking_data)
            logging.info(f"Booking information saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logging.error(f"Error saving booking: {e}")
            raise Exception(f"Error saving booking: {e}")
        
        

    def find_conversation_by_id(self, conversation_id: str):
        """
        Checks if a conversation with the given conversation_id exists in the database.

        :param conversation_id: The conversation ID to check
        :return: The conversation document if it exists, None otherwise
        """
        return self.collection.find_one({"conversation_id": conversation_id})

    async def create_new_conversation(self, record):
        try:
            self.collection.insert_one(record)
            logging.info(f"New conversation created with ID: {record['conversation_id']}")
            return str(record['conversation_id'])
        except PyMongoError as e:
            logging.error(f"Error creating new conversation: {e}")
            raise Exception(f"Error creating new conversation: {e}")

    async def append_message_to_conversation(self, conversation_id, message):
        try:
            # Retrieve the ObjectId from the conversation_id
            document = self.collection.find_one(
                {"conversation_id": conversation_id}, {"_id": 1}
            )
            if not document:
                logging.warning(
                    f"No conversation found with conversation_id: {conversation_id}"
                )
                return

            # Extract the ObjectId
            object_id = document["_id"]

            # Append the message to the conversation
            result = self.collection.update_one(
                {"_id": object_id}, {"$push": {"messages": message}}
            )

            if result.matched_count > 0:
                logging.info(
                    f"Appended message to conversation with ID: {conversation_id}"
                )
            else:
                logging.warning(f"No conversation found with ObjectId: {object_id}")
        except PyMongoError as e:
            logging.error(f"Error appending message to conversation: {e}")
            raise Exception(f"Error appending message to conversation: {e}")


    def fetch_all_queries_and_responses(self):
        results = []
        documents = self.collection.find(
            {},
            {
                "messages.user_query": 1,
                "messages.message_response": 1,
                "title": 1,
                "createdAt": 1,
                "_id": 0,
            },
        )
        for doc in documents:
            title = doc.get("title", "")
            created_at = doc.get("createdAt", "")
            messages = doc.get("messages", [])
            for message in messages:
                results.append(
                    {
                        "title": title,
                        "created_at": created_at,
                        "user_query": message["user_query"],
                        "message_response": message["message_response"],
                    }
                )
        return results