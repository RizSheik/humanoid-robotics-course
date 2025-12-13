import openai
from typing import List, Dict, Any, Optional
from backend.src.config import settings
import logging

# Set the OpenAI API key from the settings
openai.api_key = settings.openai_api_key

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI's embedding API.
        """
        try:
            # The OpenAI API processes up to 2048 texts at a time
            # For simplicity, we'll process each text individually here
            embeddings = []
            for text in texts:
                # Truncate text if it's too long (OpenAI has token limits)
                truncated_text = text[:8192]  # Conservative limit
                
                response = await self.client.embeddings.create(
                    input=truncated_text,
                    model=settings.embedding_model
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], 
                                   context: Optional[str] = None) -> str:
        """
        Generate a chat response using OpenAI's chat API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            context: Optional context to include in the conversation
            
        Returns:
            Generated response text
        """
        try:
            # Prepare the messages for the API
            prepared_messages = []
            
            # Add context if provided
            if context:
                prepared_messages.append({
                    "role": "system",
                    "content": f"Use this context when responding: {context}"
                })
            
            # Add the user's messages
            prepared_messages.extend(messages)
            
            # Make the API call
            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=prepared_messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            raise
    
    async def get_similarity_score(self, text1: str, text2: str) -> float:
        """
        Get the similarity score between two texts using embeddings.
        """
        try:
            # Generate embeddings for both texts
            embeddings = await self.get_embeddings([text1, text2])
            
            if not embeddings or len(embeddings) < 2:
                return 0.0
            
            emb1, emb2 = embeddings[0], embeddings[1]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            magnitude1 = sum(a * a for a in emb1) ** 0.5
            magnitude2 = sum(b * b for b in emb2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            return similarity
        
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

# Create a global instance of the OpenAI service
openai_service = OpenAIService()