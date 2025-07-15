import sqlite3
from logger import setup_logger
from schemas import Message, MessageHistory, User

logger = setup_logger(__name__, "database.log")

class DatabaseManager:
    DB_NAME = "users.db"

    @staticmethod
    def create_database():
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY,
                        is_finished BOOLEAN,
                        last_time_called INT,
                        query_category TEXT
                    )
                    """
                )
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        role TEXT,
                        text TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                    """
                )
                conn.commit()
                logger.info("Created users table (if not exists).")
        except Exception as e:
            logger.error(f"Failed to create users table: {e}")

    @staticmethod
    def add_user(user: User) -> bool:
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    INSERT INTO users (user_id) VALUES (?)
                """,
                    (user.user_id, ),
                )
                logger.info(f"Added user: user_id={user.user_id}")
            return True
        except Exception as e:
            logger.error(
                f"Error adding user (user_id={user.user_id}): {e}"
            )
            return False
 
    @staticmethod
    def update_last_time_called(user_id:int, last_time_called:int) -> bool:
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute('''
                    UPDATE users SET last_time_called = ? WHERE user_id = ?
                ''', (last_time_called, user_id))
                conn.commit()

                if c.rowcount == 0:
                    logger.warning(f"No user found with user_id={user_id} to update last_time_called.")
                    return False  # No row was updated
                logger.info(f"Updated last_time_called for user_id={user_id} to {last_time_called}")
                return True
        except Exception as e:
            logger.error(f"Error updating last_time_called for user_id={user_id}: {e}")
            return False
     
    @staticmethod
    def add_message(message: Message) -> bool:
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO messages (user_id, role, text) VALUES (?, ?, ?)
                ''', (message.user_id, message.role, message.text))
                conn.commit()
                logger.info(f"Message added for user_id={message.user_id}, role={message.role}")
                return True
        except Exception as e:
            logger.error(f"Error adding message for user_id={message.user_id}: {e}")
            return False

    @staticmethod
    def get_message_history(user_id: int) -> MessageHistory:
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute('''
                    SELECT role, text FROM messages WHERE user_id = ? ORDER BY message_id ASC
                ''', (user_id,))
                rows = c.fetchall()
                messages = [Message(user_id=user_id, role=role, text=text) for role, text in rows]
                logger.info(f"Retrieved {len(rows)} messages for user_id={user_id}")
                return MessageHistory(messages=messages)
        except Exception as e:
            logger.error(f"Error retrieving messages for user_id={user_id}: {e}")
            return MessageHistory(messages=[])
        
    @staticmethod
    def if_user_exists(user_id: int) -> bool:
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute('''
                    SELECT 1 FROM users WHERE user_id = ?
                ''', (user_id,))
                exists = c.fetchone() is not None
                logger.info(f"User existence check for user_id={user_id}: {exists}")
                return exists
        except Exception as e:
            logger.error(f"Error checking if user exists for user_id={user_id}: {e}")
            return False
        
    @staticmethod
    def is_user_finished(user_id: int) -> bool:
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute('''
                    SELECT is_finished FROM users WHERE user_id = ?
                ''', (user_id,))
                result = c.fetchone()
                if result is not None:
                    is_finished = bool(result[0])
                    logger.info(f"Check if user is finished for user_id={user_id}: {is_finished}")
                    return is_finished
                else:
                    logger.info(f"User with user_id={user_id} not found when checking finished status.")
                    return False
        except Exception as e:
            logger.error(f"Error checking if user is finished for user_id={user_id}: {e}")
            return False
        
    @staticmethod
    def finish_user(user_id: int) -> bool:
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute('''
                    UPDATE users SET is_finished = 1 WHERE user_id = ?
                ''', (user_id,))
                conn.commit()
                logger.info(f"Marked user_id={user_id} as finished.")
                return True
        except Exception as e:
            logger.error(f"Error marking user_id={user_id} as finished: {e}")
            return False
        
    @staticmethod
    def get_user_query_category(user_id: int) -> str | None:
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute('''
                    SELECT query_category FROM users WHERE user_id = ?
                ''', (user_id,))
                result = c.fetchone()
                if result:
                    category = result[0]
                    logger.info(f"Retrieved query_category for user_id={user_id}: {category}")
                    return category
                else:
                    logger.warning(f"No user found with user_id={user_id} when getting query_category.")
                    return None
        except Exception as e:
            logger.error(f"Error retrieving query_category for user_id={user_id}: {e}")
            return None
        
    @staticmethod
    def update_user_query_category(user_id: int, category: str) -> bool:
        try:
            with sqlite3.connect(DatabaseManager.DB_NAME) as conn:
                c = conn.cursor()
                c.execute('''
                    UPDATE users SET query_category = ? WHERE user_id = ?
                ''', (category, user_id))
                conn.commit()
                logger.info(f"Updated query_category to '{category}' for user_id={user_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating query_category for user_id={user_id}: {e}")
            return False




DatabaseManager.create_database()
