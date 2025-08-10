#!/usr/bin/env python3
"""
WebSocket handlers for streaming TTS generation.
"""
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import asyncio

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .models import (
    WebSocketRequest, WebSocketAudioResponse, WebSocketProgressResponse,
    WebSocketCompleteResponse, WebSocketErrorResponse, WebSocketStatusResponse,
    WebSocketMessageType, StreamingPerformanceMetrics
)
from .text_utils import prepare_text_for_streaming
from .tts_manager import TTSManager
from .emotion_manager import EmotionManager
from .lock_manager import GenerationLock


logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection"""
    websocket: WebSocket
    connected_at: datetime
    request_count: int = 0
    last_activity: Optional[datetime] = None


class WebSocketConnectionManager:
    """Manager for WebSocket connections and streaming TTS generation"""
    
    def __init__(self, tts_manager: TTSManager, emotion_manager: EmotionManager, 
                 generation_lock: GenerationLock):
        self.tts_manager = tts_manager
        self.emotion_manager = emotion_manager
        self.generation_lock = generation_lock
        
        # Connection tracking
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self.connection_semaphore = asyncio.Semaphore(10)  # Max concurrent connections
        
        # Performance tracking
        self.performance_metrics: List[StreamingPerformanceMetrics] = []
        
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            connection_id = str(uuid.uuid4())
            
            self.active_connections[connection_id] = ConnectionInfo(
                websocket=websocket,
                connected_at=datetime.utcnow()
            )
            
            logger.info(f"WebSocket connection established: {connection_id}")
            
            # Send connection status
            await self._send_status_update(connection_id)
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            raise
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id in self.active_connections:
            connection_info = self.active_connections[connection_id]
            duration = (datetime.utcnow() - connection_info.connected_at).total_seconds()
            
            logger.info(f"WebSocket disconnected: {connection_id} "
                       f"(duration: {duration:.2f}s, requests: {connection_info.request_count})")
            
            del self.active_connections[connection_id]
    
    async def handle_message(self, connection_id: str, message_data: dict) -> bool:
        """
        Handle incoming WebSocket message.
        
        Args:
            connection_id: WebSocket connection ID
            message_data: Parsed JSON message
            
        Returns:
            True if connection should continue, False if it should close
        """
        try:
            # Validate message format
            try:
                request = WebSocketRequest(**message_data)
            except ValidationError as e:
                await self._send_error(
                    connection_id, 
                    "Invalid request format", 
                    str(e),
                    recoverable=True
                )
                return True
            
            # Update connection activity
            if connection_id in self.active_connections:
                self.active_connections[connection_id].last_activity = datetime.utcnow()
                self.active_connections[connection_id].request_count += 1
            
            # Process the request
            await self._process_streaming_request(connection_id, request)
            return True
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {connection_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(
                connection_id,
                "Internal server error",
                str(e),
                recoverable=False
            )
            return False
    
    async def _process_streaming_request(self, connection_id: str, request: WebSocketRequest):
        """Process a streaming TTS request"""
        request_start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        
        logger.info(f"Processing streaming request {request_id} for connection {connection_id}")
        
        # Validate emotion
        emotion = self.emotion_manager.get_emotion(request.emotion)
        if not emotion:
            available_emotions = list(self.emotion_manager.list_emotions().keys())
            await self._send_error(
                connection_id,
                f"Unknown emotion: {request.emotion}",
                f"Available emotions: {', '.join(available_emotions)}",
                request_id=request_id,
                recoverable=True
            )
            return
        
        # Check if server is busy
        if self.generation_lock.is_busy:
            await self._send_status_update(connection_id, request_id, queue_info=True)
        
        # Try to acquire generation lock
        request_info = {
            "connection_id": connection_id,
            "request_id": request_id,
            "emotion": request.emotion,
            "text_length": len(request.text),
            "sentence_streaming": True,
            "start_time": datetime.utcnow().isoformat()
        }
        
        acquired, queue_time = self.generation_lock.acquire_for_generation(
            timeout=30.0,  # 30 second timeout for WebSocket requests
            request_info=request_info
        )
        
        if not acquired:
            await self._send_error(
                connection_id,
                "Server busy",
                "Too many concurrent requests. Please try again later.",
                request_id=request_id,
                recoverable=True
            )
            return
        
        try:
            # Prepare sentences for streaming
            try:
                sentences_data = prepare_text_for_streaming(request.text, max_length=1000)
            except ValueError as e:
                await self._send_error(
                    connection_id,
                    "Text validation failed",
                    str(e),
                    request_id=request_id,
                    recoverable=True
                )
                return
            
            if not sentences_data:
                await self._send_error(
                    connection_id,
                    "No valid sentences found",
                    "Text could not be processed into sentences",
                    request_id=request_id,
                    recoverable=True
                )
                return
            
            # Send initial progress update
            if request.include_progress:
                await self._send_progress_update(
                    connection_id, request_id, 0, sentences_data[0]['text'],
                    len(sentences_data), 0.0
                )
            
            # Generate audio using sentence streaming for voice consistency
            total_duration = 0.0
            sentence_metrics = []
            time_to_first_sentence = None
            
            logger.info(f"Starting sentence-based streaming generation for {len(sentences_data)} sentences")
            
            # Prepare reference audio conditioning if provided
            if request.reference_audio:
                try:
                    from pathlib import Path
                    audio_path = Path(request.reference_audio)
                    if not audio_path.exists():
                        await self._send_error(
                            connection_id,
                            "Reference audio not found",
                            f"File not found: {request.reference_audio}",
                            request_id=request_id,
                            recoverable=True
                        )
                        return
                    
                    # Prepare conditionals with reference audio
                    logger.info(f"Using reference audio: {audio_path.name}")
                    self.tts_manager.prepare_conditionals(str(audio_path))
                    
                except Exception as e:
                    await self._send_error(
                        connection_id,
                        "Failed to load reference audio",
                        str(e),
                        request_id=request_id,
                        recoverable=True
                    )
                    return
            
            # Use the streaming generator that maintains voice consistency
            sentence_generator = self.tts_manager.generate_sentences_stream(
                sentences_data=sentences_data,
                emotion=request.emotion,
                temperature=request.temperature,
                cfg_weight=request.cfg_weight,
                exaggeration=request.exaggeration
            )
            
            try:
                for sentence_index, sentence_text, audio_tensor, duration, generation_time, voice_sample in sentence_generator:
                    logger.info(f"Completed sentence {sentence_index + 1}/{len(sentences_data)}: "
                              f"'{sentence_text[:50]}{'...' if len(sentence_text) > 50 else ''}'")
                    
                    # Convert to base64
                    audio_base64 = self.tts_manager.audio_to_base64(audio_tensor)
                    
                    # Calculate metrics
                    rtf = generation_time / duration if duration > 0 else 0
                    total_duration += duration
                    
                    # Record time to first sentence
                    if time_to_first_sentence is None:
                        time_to_first_sentence = time.time() - request_start_time
                    
                    # Store sentence metrics
                    sentence_metric = {
                        'index': sentence_index,
                        'text': sentence_text,
                        'duration': duration,
                        'generation_time': generation_time,
                        'rtf': rtf,
                        'voice_sample': voice_sample
                    }
                    sentence_metrics.append(sentence_metric)
                    
                    # Send audio chunk
                    await self._send_audio_chunk(
                        connection_id=connection_id,
                        request_id=request_id,
                        sentence_index=sentence_index,
                        sentence_text=sentence_text,
                        audio_base64=audio_base64,
                        duration=duration,
                        generation_time=generation_time,
                        rtf=rtf,
                        is_final=(sentence_index == len(sentences_data) - 1),
                        cumulative_duration=total_duration
                    )
                    
                    # Send progress update
                    if request.include_progress and sentence_index < len(sentences_data) - 1:
                        progress_percent = ((sentence_index + 1) / len(sentences_data)) * 100
                        next_sentence_text = sentences_data[sentence_index + 1]['text'] \
                            if sentence_index + 1 < len(sentences_data) else ""
                        
                        await self._send_progress_update(
                            connection_id, request_id, sentence_index + 1,
                            next_sentence_text, len(sentences_data), progress_percent
                        )
                    
                    logger.info(f"Sentence {sentence_index + 1} completed: "
                              f"{duration:.2f}s audio, {generation_time:.2f}s gen time, RTF: {rtf:.3f}")
                
            except Exception as generation_error:
                logger.error(f"Error in sentence streaming generation: {generation_error}")
                
                await self._send_error(
                    connection_id,
                    "Streaming generation failed",
                    str(generation_error),
                    request_id=request_id,
                    recoverable=False
                )
                return
            
            # Send completion message
            total_generation_time = time.time() - request_start_time
            overall_rtf = total_generation_time / total_duration if total_duration > 0 else 0
            
            await self._send_completion(
                connection_id=connection_id,
                request_id=request_id,
                total_sentences=len(sentences_data),
                total_duration=total_duration,
                total_generation_time=total_generation_time,
                overall_rtf=overall_rtf,
                time_to_first_sentence=time_to_first_sentence or 0.0,
                emotion_used=request.emotion,
                text_normalized=request.text
            )
            
            # Store performance metrics
            metrics = StreamingPerformanceMetrics(
                request_id=request_id,
                text_length=len(request.text),
                sentence_count=len(sentences_data),
                time_to_first_sentence=time_to_first_sentence or 0.0,
                total_generation_time=total_generation_time,
                total_audio_duration=total_duration,
                overall_rtf=overall_rtf,
                average_sentence_generation_time=sum(m['generation_time'] for m in sentence_metrics) / len(sentence_metrics),
                sentence_metrics=sentence_metrics,
                emotion_used=request.emotion,
                timestamp=datetime.utcnow().isoformat()
            )
            self.performance_metrics.append(metrics)
            
            logger.info(f"Streaming request {request_id} completed: "
                       f"{len(sentences_data)} sentences, {total_duration:.2f}s audio, "
                       f"TTFS: {time_to_first_sentence:.3f}s, Overall RTF: {overall_rtf:.3f}")
        
        finally:
            self.generation_lock.release()
    
    async def _send_audio_chunk(self, connection_id: str, request_id: Optional[str],
                               sentence_index: int, sentence_text: str, audio_base64: str,
                               duration: float, generation_time: float, rtf: float,
                               is_final: bool, cumulative_duration: float):
        """Send an audio chunk to the client"""
        response = WebSocketAudioResponse(
            request_id=request_id,
            sentence_index=sentence_index,
            sentence_text=sentence_text,
            audio_base64=audio_base64,
            duration=duration,
            generation_time=generation_time,
            rtf=rtf,
            is_final=is_final,
            cumulative_duration=cumulative_duration
        )
        await self._send_message(connection_id, response.dict())
    
    async def _send_progress_update(self, connection_id: str, request_id: Optional[str],
                                   sentence_index: int, sentence_text: str,
                                   total_sentences: int, progress_percent: float,
                                   estimated_remaining_time: Optional[float] = None):
        """Send a progress update to the client"""
        response = WebSocketProgressResponse(
            request_id=request_id,
            sentence_index=sentence_index,
            sentence_text=sentence_text,
            total_sentences=total_sentences,
            progress_percent=progress_percent,
            estimated_remaining_time=estimated_remaining_time
        )
        await self._send_message(connection_id, response.dict())
    
    async def _send_completion(self, connection_id: str, request_id: Optional[str],
                              total_sentences: int, total_duration: float,
                              total_generation_time: float, overall_rtf: float,
                              time_to_first_sentence: float, emotion_used: str,
                              text_normalized: str):
        """Send a completion message to the client"""
        response = WebSocketCompleteResponse(
            request_id=request_id,
            total_sentences=total_sentences,
            total_duration=total_duration,
            total_generation_time=total_generation_time,
            overall_rtf=overall_rtf,
            time_to_first_sentence=time_to_first_sentence,
            emotion_used=emotion_used,
            text_normalized=text_normalized
        )
        await self._send_message(connection_id, response.dict())
    
    async def _send_error(self, connection_id: str, error: str, detail: Optional[str] = None,
                         request_id: Optional[str] = None, sentence_index: Optional[int] = None,
                         recoverable: bool = False):
        """Send an error message to the client"""
        response = WebSocketErrorResponse(
            request_id=request_id,
            error=error,
            detail=detail,
            sentence_index=sentence_index,
            recoverable=recoverable
        )
        await self._send_message(connection_id, response.dict())
    
    async def _send_status_update(self, connection_id: str, request_id: Optional[str] = None,
                                 queue_info: bool = False):
        """Send a status update to the client"""
        response = WebSocketStatusResponse(
            connected=True,
            server_busy=self.generation_lock.is_busy,
            queue_position=None,  # Could implement queue position tracking
            estimated_wait_time=None
        )
        await self._send_message(connection_id, response.dict())
    
    async def _send_message(self, connection_id: str, message_dict: dict):
        """Send a message to a specific WebSocket connection"""
        if connection_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent connection: {connection_id}")
            return
        
        connection_info = self.active_connections[connection_id]
        
        try:
            await connection_info.websocket.send_text(json.dumps(message_dict))
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {e}")
            # Remove the connection as it's likely dead
            await self.disconnect(connection_id)
    
    def get_connection_stats(self) -> dict:
        """Get statistics about active connections"""
        now = datetime.utcnow()
        
        stats = {
            'active_connections': len(self.active_connections),
            'total_requests_processed': sum(conn.request_count for conn in self.active_connections.values()),
            'connections': []
        }
        
        for conn_id, conn_info in self.active_connections.items():
            uptime = (now - conn_info.connected_at).total_seconds()
            last_activity = (now - conn_info.last_activity).total_seconds() \
                if conn_info.last_activity else uptime
            
            stats['connections'].append({
                'connection_id': conn_id,
                'uptime_seconds': uptime,
                'last_activity_seconds': last_activity,
                'request_count': conn_info.request_count
            })
        
        return stats
    
    def get_performance_metrics(self, limit: int = 50) -> List[dict]:
        """Get recent performance metrics"""
        return [metric.dict() for metric in self.performance_metrics[-limit:]]