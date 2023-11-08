| Method                        | Description                                         | Access Level | Required Keys in Request Dictionary               |
|-------------------------------|-----------------------------------------------------|--------------|---------------------------------------------------|
| `query_inference`             | Send a prompt to a LLM for inference                  | User         | `prompt`(srt), `memory`(bool), `conversation_number` (int), `AI_assistance` (bool),  `collection_name` (str)|
| `add_user`                    | Adds a new user to the system.                      | Admin        | `username`, `password`, `token_limit` (Optional)  |
| `update_token_limit`          | Updates the token limit for a user.                 | Admin        | `username`, `token_limit`              |
| `get_all_data`                | Retrieves all data from the database.               | Admin        | N/A                                               |
| `delete_user`                 | Deletes a user and all related content.             | Admin        | `username`                                        |
| `check_user_existence`        | Checks if a user exists in the database.            | Admin        | `username`                                        |
| `disable_user`                | Disables a user account.                            | Admin        | `username`                                        |
| `add_conversation`            | Adds a conversation to the database.                | User         | `content`, `conversation_name`                    |
| `delete_conversation`         | Deletes a conversation from the database.           | User         | `conversation_number`                             |
| `retrieve_conversation`       | Retrieves a conversation from the database.         | User         | `conversation_number`                             |
| `retrieve_latest_conversation`| Retrieves the latest conversation.                  | User         | N/A                                               |
| `update_conversation`         | Updates a conversation in the database.             | User         | `conversation_number`, `content`                  |
| `update_conversation_name`    | Updates the name of a conversation in the database. | User         | `conversation_number`, `new_name`                 |
| `get_user_conversations`      | Retrieves all conversations for a user.             | User         | N/A                                               |
