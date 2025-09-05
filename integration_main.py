#!/usr/bin/env python3
"""
AstraDB Integration for Components 2+3+4
Combines emotion analysis, semantic analysis, and feature engineering
to produce outputs formatted for AstraDB collections
"""

import sys
import uuid
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
import logging
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add AstraDB imports
try:
    from astrapy import DataAPIClient
    from astrapy.exceptions import TableInsertManyException
    logger.info("✅ AstraDB client imported successfully")
except ImportError as e:
    logger.error(f"❌ AstraDB client import failed: {e}")
    logger.info("Install with: pip install astrapy")
    raise

# Add component paths to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
COMP2_PATH = PROJECT_ROOT / "comp2"
COMP3_PATH = PROJECT_ROOT / "comp3"
COMP4_PATH = PROJECT_ROOT / "comp4"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(COMP2_PATH))
sys.path.insert(0, str(COMP3_PATH))
sys.path.insert(0, str(COMP4_PATH))

# Import components
try:
    from comp2.src.emotion_analyzer import EmotionAnalyzer
    from comp2.data.schemas import EmotionAnalysis, EmotionScores
    logger.info("✅ Component 2 imported successfully")
except ImportError as e:
    logger.error(f"❌ Component 2 import failed: {e}")
    raise

try:
    from comp3.src.analyzer import Component3Analyzer
    from comp3.data.schemas import SemanticAnalysis
    from comp3.src.event_extractor import EventExtractor
    logger.info("✅ Component 3 imported successfully")
except ImportError as e:
    logger.error(f"❌ Component 3 import failed: {e}")
    raise

try:
    from comp4.src.processor import Component4Processor
    from comp4.data.schemas import EngineeredFeatures, UserHistoryContext
    logger.info("✅ Component 4 imported successfully")
except ImportError as e:
    logger.error(f"❌ Component 4 import failed: {e}")
    raise

@dataclass
class AstraDBOutput:
    """Formatted output for AstraDB storage"""
    
    # Collection 1: chat_embeddings
    chat_embeddings: Dict[str, Any]
    
    # Collection 2: semantic_search
    semantic_search: Dict[str, Any]
    
    # Processing metadata
    processing_time_ms: float
    component_versions: Dict[str, str]
    timestamp: datetime

class AstraDBConnector:
    """HTTP client for chat embeddings and semantic search endpoints"""
    
    def __init__(self):
        # Get the endpoints from environment variables
        self.endpoints = {
            "chat_embeddings": os.getenv("CHAT_EMBEDDINGS_COLLECTION_ENDPOINT"),
            "semantic_search": os.getenv("SEMANTIC_SEARCH_COLLECTION_ENDPOINT")
        }
        
        # Validate endpoints
        for name, endpoint in self.endpoints.items():
            if not endpoint:
                raise ValueError(f"Missing {name.upper()}_COLLECTION_ENDPOINT environment variable")
            
            # Ensure the endpoint ends with a slash
            if not endpoint.endswith('/'):
                self.endpoints[name] = endpoint + '/'
        
        logger.info("✅ Endpoints configured for chat_embeddings and semantic_search")
    
    def push_to_collection(self, collection_name: str, data: Dict[str, Any]) -> bool:
        """Push data to the specified collection via HTTP endpoint"""
        if collection_name not in self.endpoints:
            logger.warning(f"Unsupported collection: {collection_name}. Must be one of: {list(self.endpoints.keys())}")
            return False
            
        try:
            import requests
            if collection_name == "semantic_search":
                endpoint = self.endpoints[collection_name]

                # # Check if the entry already exists in the collection by using the entry_id, if it exists give a put request request
                # response = requests.get(
                #     f"{endpoint}api/semantic-search/entries/{data['entry_id']}",
                #     headers={"Content-Type": "application/json"},
                #     timeout=10  # 10 second timeout
                # )
                # if response.status_code == 200:
                #     logger.info(f"✅ Entry {data['entry_id']} already exists in the collection, updating it")
                #     response = requests.put(
                #         f"{endpoint}api/semantic-search/entries/{data['entry_id']}",
                #         json=data,
                #         headers={"Content-Type": "application/json"},
                #         timeout=10  # 10 second timeout
                #     )
                # else:
                    # Send POST request to the appropriate endpoint
                response = requests.post(
                    f"{endpoint}api/semantic-search/entries",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=10  # 10 second timeout
                )
            elif collection_name == "chat_embeddings":
                endpoint = self.endpoints[collection_name]
                # Send POST request to the appropriate endpoint
                # response = requests.get(
                #     f"{endpoint}api/chat-embeddings/entries/{data['entry_id']}",
                #     headers={"Content-Type": "application/json"},
                #     timeout=10  # 10 second timeout
                # )
                # if response.status_code == 200:
                #     logger.info(f"✅ Entry {data['entry_id']} already exists in the collection, updating it")
                #     response = requests.put(
                #         f"{endpoint}api/chat-embeddings/{data['entry_id']}",
                #         json=data,
                #         headers={"Content-Type": "application/json"},
                #         timeout=10  # 10 second timeout
                #     )
                # else:
                    # Send POST request to the appropriate endpoint
                response = requests.post(
                    f"{endpoint}api/chat-embeddings",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=10  # 10 second timeout
                )

            if response.status_code == 201:
                logger.info(f"✅ Data pushed to {collection_name} collection successfully")
                return True
            else:
                logger.error(f"❌ Failed to push to {collection_name}: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error(f"❌ Timeout while pushing to {collection_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Error pushing to {collection_name}: {str(e)}")
            return False

class AstraDBIntegrator:
    """
    Integrates Components 2+3+4 to produce AstraDB-formatted outputs
    """
    
    def __init__(self, config_path: str = "unified_config.yaml"):
        """Initialize the integrator with all components"""
        self.config_path = config_path
        
        # Initialize AstraDB connector
        self.astra_connector = AstraDBConnector()
        
        # Initialize Component 2: Emotion Analysis
        self.emotion_analyzer = EmotionAnalyzer()
        
        # Initialize Component 3: Semantic Analysis
        self.semantic_analyzer = Component3Analyzer()
        
        # Initialize Event Extractor for temporal events
        self.event_extractor = EventExtractor()
        
        # Initialize Component 4: Feature Engineering
        self.feature_processor = Component4Processor()
        
        logger.info("✅ AstraDB Integrator initialized with all components")
    
    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Check if string is a valid UUID"""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    

    def process_journal_entry(
        self,
        text: str,
        user_id: str,
        session_id: str = None,
        entry_timestamp: datetime = None,
        entry_id: str = None,
        message_type: str = "user_message",
        user_history: Optional[UserHistoryContext] = None
    ) -> AstraDBOutput:
        """
        Process journal entry through complete pipeline and format for AstraDB
        
        Args:
            text: Journal entry text
            user_id: User identifier
            session_id: Session identifier
            entry_timestamp: Entry timestamp
            entry_id: Entry identifier
            message_type: Type of message (user_message, ai_response, system_message)
            user_history: User's historical context for personalized feature engineering
            
        Returns:
            AstraDBOutput with formatted data for both collections
        """
        start_time = time.time()
        
        # Set defaults
        entry_timestamp = entry_timestamp or datetime.now()
        entry_id = entry_id or str(uuid.uuid4())
        # Generate a proper UUID for session_id if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        elif not self._is_valid_uuid(session_id):
            # If provided session_id is not a valid UUID, generate one
            session_id = str(uuid.uuid4())

        
        logger.info(f"🔄 Processing entry {entry_id} for user {user_id}")
        
        try:
            # Step 1: Component 2 - Emotion Analysis
            logger.info("📊 Processing emotion analysis (Component 2)...")
            emotion_result = self.emotion_analyzer.analyze_emotion(
                text=text,
                user_id=user_id
            )
            
            # Step 2: Component 3 - Semantic Analysis
            logger.info("🔍 Processing semantic analysis (Component 3)...")
            semantic_result = self.semantic_analyzer.analyze(
                processed_text=text,
                user_id=user_id,
                entry_timestamp=entry_timestamp
            )
            
            # Step 3: Extract and Store Events
            logger.info("📅 Extracting and storing temporal events...")
            event_result = self.event_extractor.extract_and_store_events(
                text=text,
                user_id=user_id,
                reference_date=entry_timestamp
            )
            logger.info(f"📅 Events processed: {event_result.get('events_extracted', 0)} extracted, {event_result.get('events_stored', 0)} stored")
            
            # Step 4: Component 4 - Feature Engineering
            logger.info("⚙️ Processing feature engineering (Component 4)...")
            
            # Create journal entry for Component 4
            journal_entry = {
                'entry_id': entry_id,
                'user_id': user_id,
                'content': text,
                'timestamp': entry_timestamp,
                'session_id': session_id,
                'emotion_analysis': emotion_result,
                'semantic_analysis': semantic_result,
                'user_history': user_history
            }
            
            engineered_features = self.feature_processor.process_journal_entry(
                emotion_analysis=emotion_result,
                semantic_analysis=semantic_result,
                user_id=user_id,
                entry_id=entry_id,
                session_id=session_id,
                entry_timestamp=entry_timestamp,
                raw_text=text,
                user_history=user_history
            )
            
            # Step 5: Format for AstraDB
            logger.info("🗄️ Formatting for AstraDB...")
            chat_embeddings = self._format_chat_embeddings(
                text=text,
                user_id=user_id,
                entry_id=entry_id,
                session_id=session_id,
                entry_timestamp=entry_timestamp,
                message_type=message_type,
                emotion_analysis=emotion_result,
                semantic_analysis=semantic_result,
                engineered_features=engineered_features
            )
            
            semantic_search = self._format_semantic_search(
                text=text,
                user_id=user_id,
                entry_id=entry_id,
                session_id=session_id,
                entry_timestamp=entry_timestamp,
                emotion_analysis=emotion_result,
                semantic_analysis=semantic_result,
                engineered_features=engineered_features
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create output
            output = AstraDBOutput(
                chat_embeddings=chat_embeddings,
                semantic_search=semantic_search,
                processing_time_ms=processing_time,
                component_versions={
                    "component2": "2.0",
                    "component3": "3.0", 
                    "component4": "4.0"
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"✅ Processing completed in {processing_time:.1f}ms")
            return output
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise

    def push_to_astra_db(self, output: AstraDBOutput) -> bool:
        """
        Push the processed data to AstraDB collections
        
        Args:
            output: AstraDBOutput containing formatted data
            
        Returns:
            bool: True if both pushes successful, False otherwise
        """
        logger.info("🚀 Pushing data to AstraDB collections...")
        
        try:
            # Push to chat_embeddings collection
            chat_success = self.astra_connector.push_to_collection(
                "chat_embeddings", 
                output.chat_embeddings
            )
            
            # Push to semantic_search collection
            search_success = self.astra_connector.push_to_collection(
                "semantic_search", 
                output.semantic_search
            )
            
            if chat_success and search_success:
                logger.info("✅ Successfully pushed data to both collections")
                return True
            else:
                logger.error("❌ Failed to push to one or both collections")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error pushing to AstraDB: {e}")
            return False

    # [Keep all the existing formatting methods unchanged]
    def _format_chat_embeddings(
        self,
        text: str,
        user_id: str,
        entry_id: str,
        session_id: str,
        entry_timestamp: datetime,
        message_type: str,
        emotion_analysis: EmotionAnalysis,
        semantic_analysis: SemanticAnalysis,
        engineered_features: EngineeredFeatures
    ) -> Dict[str, Any]:
        """Format data for chat_embeddings collection - STRICTLY following the provided schema"""
        
        # Extract embeddings from Component 3 and ensure they are proper float arrays
        primary_embedding = semantic_analysis.embeddings.primary_embedding
        lightweight_embedding = semantic_analysis.embeddings.lightweight_embedding
        
        # Convert to proper float arrays and ensure correct dimensions
        if len(primary_embedding) != 768:
            logger.warning(f"Primary embedding dimension mismatch: {len(primary_embedding)} != 768")
            primary_embedding = self._pad_or_truncate_embedding(primary_embedding, 768)
        
        if len(lightweight_embedding) != 384:
            logger.warning(f"Lightweight embedding dimension mismatch: {len(lightweight_embedding)} != 384")
            lightweight_embedding = self._pad_or_truncate_embedding(lightweight_embedding, 384)
        
        # CRITICAL: Convert to proper float arrays for AstraDB vector operations
        primary_embedding = [float(x) for x in primary_embedding]
        lightweight_embedding = [float(x) for x in lightweight_embedding]
        
        # Format emotion context - EXACTLY as per schema
        emotion_context = {
            "dominant_emotion": emotion_analysis.dominant_emotion,
            "intensity": float(emotion_analysis.intensity),
            "emotions": {
                "joy": float(emotion_analysis.emotions.joy),
                "sadness": float(emotion_analysis.emotions.sadness),
                "anger": float(emotion_analysis.emotions.anger),
                "fear": float(emotion_analysis.emotions.fear),
                "surprise": float(emotion_analysis.emotions.surprise),
                "disgust": float(emotion_analysis.emotions.disgust),
                "anticipation": float(emotion_analysis.emotions.anticipation),
                "trust": float(emotion_analysis.emotions.trust)
            }
        }
        
        # Format entities - EXACTLY as per schema
        entities_mentioned = {
            "people": [p.name for p in semantic_analysis.people],
            "locations": [l.name for l in semantic_analysis.locations],
            "organizations": [o.name for o in semantic_analysis.organizations]
        }
        
        # Format temporal context - EXACTLY as per schema
        temporal_context = {
            "hour_of_day": int(entry_timestamp.hour),
            "day_of_week": int(entry_timestamp.weekday()),
            "is_weekend": bool(entry_timestamp.weekday() >= 5)
        }
        
        # Extract semantic tags from Component 4
        semantic_tags = getattr(engineered_features.metadata, 'retrieval_triggers', []) if hasattr(engineered_features, 'metadata') else []
        
        # Extract Component 4 feature vectors and quality metrics
        feature_vector_90d = getattr(engineered_features, 'feature_vector', [])
        temporal_features_25d = getattr(engineered_features, 'temporal_features', [])
        emotional_features_20d = getattr(engineered_features, 'emotional_features', [])
        semantic_features_30d = getattr(engineered_features, 'semantic_features', [])
        user_features_15d = getattr(engineered_features, 'user_features', [])
        
        # Extract quality metrics
        feature_completeness = getattr(engineered_features, 'feature_completeness', 0.0)
        confidence_score = getattr(engineered_features, 'confidence_score', 0.0)
        
        # Generate conversation context
        conversation_context = f"Journal entry by {user_id} discussing {', '.join(semantic_tags[:3]) if semantic_tags else 'various topics'}"
        
        # STRICTLY follow the exact schema provided
        return {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "entry_id": entry_id,
            "message_content": text,  # EXACT field name as per schema
            "message_type": message_type,
            "timestamp": entry_timestamp.isoformat() + "Z",  # ISO format with Z suffix as per schema
            "session_id": session_id,
            "conversation_context": conversation_context,
            "primary_embedding": primary_embedding,  # 768 dimensions - PROPER FLOAT ARRAY
            "lightweight_embedding": lightweight_embedding,  # 384 dimensions - PROPER FLOAT ARRAY
            "text_length": int(len(text)),  # integer as per schema
            "processing_time_ms": float(engineered_features.processing_time_ms),  # float as per schema
            "model_version": f"C2:{emotion_analysis.model_version}, C3:3.0, C4:4.0",
            "semantic_tags": semantic_tags,
            "emotion_context": emotion_context,
            "entities_mentioned": entities_mentioned,
            "temporal_context": temporal_context,
            
            # ADDITIONAL COMPONENT 4 FEATURES (Enhanced functionality)
            "feature_vector": self._safe_convert_to_float_list(feature_vector_90d),
            "temporal_features": self._safe_convert_to_float_list(temporal_features_25d),
            "emotional_features": self._safe_convert_to_float_list(emotional_features_20d),
            "semantic_features": self._safe_convert_to_float_list(semantic_features_30d),
            "user_features": self._safe_convert_to_float_list(user_features_15d),
            "feature_completeness": float(feature_completeness),
            "confidence_score": float(confidence_score)
        }
    
    def _format_semantic_search(
        self,
        text: str,
        user_id: str,
        entry_id: str,
        session_id: str,
        entry_timestamp: datetime,
        emotion_analysis: EmotionAnalysis,
        semantic_analysis: SemanticAnalysis,
        engineered_features: EngineeredFeatures
    ) -> Dict[str, Any]:
        """Format data for semantic_search collection - STRICTLY following the provided schema"""
        
        # Extract primary embedding and ensure it's a proper float array
        primary_embedding = semantic_analysis.embeddings.primary_embedding
        if len(primary_embedding) != 768:
            primary_embedding = self._pad_or_truncate_embedding(primary_embedding, 768)
        
        # CRITICAL: Convert to proper float array for AstraDB vector operations
        primary_embedding = [float(x) for x in primary_embedding]
        
        # Determine content type based on analysis
        content_type = self._determine_content_type(
            emotion_analysis, semantic_analysis, engineered_features
        )
        
        # Generate title from content
        title = self._generate_title(text, content_type)
        
        # Extract tags and Component 4 features
        tags = getattr(engineered_features.metadata, 'retrieval_triggers', []) if hasattr(engineered_features, 'metadata') else []
        
        # Extract Component 4 feature vectors for enhanced search capabilities
        feature_vector_90d = getattr(engineered_features, 'feature_vector', [])
        temporal_features_25d = getattr(engineered_features, 'temporal_features', [])
        emotional_features_20d = getattr(engineered_features, 'emotional_features', [])
        semantic_features_30d = getattr(engineered_features, 'semantic_features', [])
        user_features_15d = getattr(engineered_features, 'user_features', [])
        
        # Format linked entities - EXACTLY as per schema
        linked_entities = {
            "people": [p.name for p in semantic_analysis.people],
            "locations": [l.name for l in semantic_analysis.locations],
            "events": [e.event_text for e in semantic_analysis.future_events],
            "topics": tags
        }
        
        # Calculate search metadata - EXACTLY as per schema
        search_metadata = {
            "boost_factor": float(engineered_features.metadata.importance_score if hasattr(engineered_features.metadata, 'importance_score') else 0.5),
            "recency_weight": float(1.0),  # Can be adjusted based on business logic
            "user_preference_alignment": float(engineered_features.metadata.confidence if hasattr(engineered_features.metadata, 'confidence') else 0.5)
        }
        
        # STRICTLY follow the exact schema provided
        return {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "entry_id": entry_id,
            "session_id": session_id,
            "content_type": content_type,
            "title": title,
            "content": text,  # EXACT field name as per schema
            "primary_embedding": primary_embedding,  # 768 dimensions - PROPER FLOAT ARRAY
            "created_at": entry_timestamp.isoformat() + "Z",  # ISO format with Z suffix as per schema
            "updated_at": entry_timestamp.isoformat() + "Z",  # ISO format with Z suffix as per schema
            "tags": tags,
            "linked_entities": linked_entities,
            "search_metadata": search_metadata,
            
            # ADDITIONAL COMPONENT 4 FEATURES (Enhanced search capabilities)
            "feature_vector": self._safe_convert_to_float_list(feature_vector_90d),
            "temporal_features": self._safe_convert_to_float_list(temporal_features_25d),
            "emotional_features": self._safe_convert_to_float_list(emotional_features_20d),
            "semantic_features": self._safe_convert_to_float_list(semantic_features_30d),
            "user_features": self._safe_convert_to_float_list(user_features_15d)
        }
    
    def _pad_or_truncate_embedding(self, embedding: List[float], target_dim: int) -> List[float]:
        """Ensure embedding has correct dimensions and proper float format"""
        # Convert to proper float array first
        float_embedding = [float(x) for x in embedding]
        
        if len(float_embedding) > target_dim:
            return float_embedding[:target_dim]
        elif len(float_embedding) < target_dim:
            return float_embedding + [0.0] * (target_dim - len(float_embedding))
        return float_embedding
    
    def _safe_convert_to_float_list(self, feature_vector) -> List[float]:
        """Safely convert numpy arrays or other iterables to float lists"""
        if feature_vector is None:
            return []
        
        try:
            # Handle numpy arrays
            if hasattr(feature_vector, 'tolist'):
                return [float(x) for x in feature_vector.tolist()]
            # Handle regular lists/iterables
            elif hasattr(feature_vector, '__iter__') and not isinstance(feature_vector, str):
                return [float(x) for x in feature_vector]
            else:
                return []
        except (ValueError, TypeError):
            return []
    
    def _determine_content_type(
        self,
        emotion_analysis: EmotionAnalysis,
        semantic_analysis: SemanticAnalysis,
        engineered_features: EngineeredFeatures
    ) -> str:
        """Determine the content type based on analysis results"""
        
        # Check for events
        if semantic_analysis.future_events:
            return "event"
        
        # Check for people mentions
        if semantic_analysis.people:
            return "person"
        
        # Check for location mentions
        if semantic_analysis.locations:
            return "location"
        
        # Check for topic diversity
        if hasattr(engineered_features.metadata, 'retrieval_triggers'):
            if len(engineered_features.metadata.retrieval_triggers) > 3:
                return "topic"
        
        # Default to journal entry
        return "journal_entry"
    
    def _generate_title(self, text: str, content_type: str) -> str:
        """Generate a title for the content"""
        if content_type == "event":
            return f"Event: {text[:50]}..."
        elif content_type == "person":
            return f"Person Mention: {text[:50]}..."
        elif content_type == "location":
            return f"Location: {text[:50]}..."
        elif content_type == "topic":
            return f"Topic Discussion: {text[:50]}..."
        else:
            return f"Journal Entry: {text[:50]}..."
    
    def batch_process(
        self,
        entries: List[Dict[str, Any]]
    ) -> List[AstraDBOutput]:
        """Process multiple journal entries in batch"""
        results = []
        session_id = str(uuid.uuid4())
        for entry in entries:
            try:
                result = self.process_journal_entry(
                    text=entry["text"],
                    user_id=entry["user_id"],
                    session_id=session_id,
                    entry_timestamp=entry.get("entry_timestamp"),
                    entry_id=entry.get("entry_id"),
                    message_type=entry.get("message_type", "user_message")
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process entry: {e}")
                continue
        
        return results
    
    def export_for_astra_db(self, output: AstraDBOutput) -> Dict[str, Any]:
        """Export formatted data ready for AstraDB insertion"""
        return {
            "chat_embeddings": output.chat_embeddings,
            "semantic_search": output.semantic_search,
            "metadata": {
                "processing_time_ms": output.processing_time_ms,
                "component_versions": output.component_versions,
                "timestamp": output.timestamp.isoformat()
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize integrator
    integrator = AstraDBIntegrator()
    
    # Create sample user history context
    user_history = UserHistoryContext(
        writing_frequency_baseline=0.75,
        emotional_baseline={
            "joy": 0.6,
            "sadness": 0.2,
            "anger": 0.1,
            "fear": 0.1,
            "surprise": 0.3,
            "disgust": 0.05,
            "anticipation": 0.4,
            "trust": 0.7
        },
        topic_preferences=["personal_growth", "relationships", "work", "health"],
        behavioral_patterns={
            "writing_style": "reflective",
            "session_length": "medium",
            "emotional_expression": "moderate"
        },
        last_entry_timestamp=datetime.now(),
        total_entries=25,
        avg_session_duration=15.5,
        preferred_writing_times=[9, 14, 21],
        emotional_volatility=0.3,
        topic_consistency=0.8,
        social_connectivity=0.6
    )
    
    # Test with sample journal entry
    sample_text = "I have an event at Google on 2nd of September with my friend Sarah."
    
    try:
        # Process the journal entry
        result = integrator.process_journal_entry(
            text=sample_text,
            user_id="550e8400-e29b-41d4-a716-446655440000",
            session_id="session-456",
            user_history=user_history
        )
        
        # Push to AstraDB collections
        success = integrator.push_to_astra_db(result)
        
        if success:
            print("✅ Integration and AstraDB push successful!")
            print(f"Processing time: {result.processing_time_ms:.1f}ms")
            print(f"Chat embeddings ID: {result.chat_embeddings['id']}")
            print(f"Semantic search ID: {result.semantic_search['id']}")
        else:
            print("❌ Failed to push data to AstraDB")
        # Export for AstraDB (optional - for debugging)
        astra_data = integrator.export_for_astra_db(result)
        
        # Save sample output
        with open("astra_db_sample_output.json", "w") as f:
            json.dump(astra_data, f, indent=2, default=str)
        print("📁 Sample output saved to astra_db_sample_output.json")
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
