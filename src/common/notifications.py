"""Notifications for ML pipeline events (Slack, Telegram, Email)."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from src.common.logging import get_logger

logger = get_logger(__name__)


class NotificationLevel(Enum):
    """Notification severity levels."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Notification:
    """Notification message."""
    title: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class NotificationChannel(ABC):
    """Abstract notification channel."""
    
    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send notification. Returns True on success."""
        pass


class SlackChannel(NotificationChannel):
    """Slack webhook notification channel."""
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        channel: Optional[str] = None,
        username: str = "Churn Radar",
        icon_emoji: str = ":robot_face:",
    ):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
    
    def _get_color(self, level: NotificationLevel) -> str:
        """Get Slack attachment color for level."""
        colors = {
            NotificationLevel.INFO: "#2196F3",
            NotificationLevel.SUCCESS: "#4CAF50",
            NotificationLevel.WARNING: "#FF9800",
            NotificationLevel.ERROR: "#F44336",
        }
        return colors.get(level, "#757575")
    
    def _build_payload(self, notification: Notification) -> Dict[str, Any]:
        """Build Slack message payload."""
        attachment = {
            "color": self._get_color(notification.level),
            "title": notification.title,
            "text": notification.message,
            "ts": datetime.utcnow().timestamp(),
            "footer": "Churn Radar ML Platform",
        }
        
        if notification.metadata:
            fields = []
            for key, value in notification.metadata.items():
                fields.append({
                    "title": key,
                    "value": str(value),
                    "short": True,
                })
            attachment["fields"] = fields
        
        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [attachment],
        }
        
        if self.channel:
            payload["channel"] = self.channel
        
        return payload
    
    async def send(self, notification: Notification) -> bool:
        """Send notification to Slack."""
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        payload = self._build_payload(notification)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                
                if response.status_code == 200:
                    logger.info(f"Slack notification sent: {notification.title}")
                    return True
                else:
                    logger.error(f"Slack error: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False


class TelegramChannel(NotificationChannel):
    """Telegram bot notification channel."""
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    
    def _get_emoji(self, level: NotificationLevel) -> str:
        """Get emoji for notification level."""
        emojis = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.SUCCESS: "✅",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
        }
        return emojis.get(level, "📢")
    
    def _build_message(self, notification: Notification) -> str:
        """Build Telegram message text."""
        emoji = self._get_emoji(notification.level)
        
        lines = [
            f"{emoji} *{notification.title}*",
            "",
            notification.message,
        ]
        
        if notification.metadata:
            lines.append("")
            for key, value in notification.metadata.items():
                lines.append(f"• *{key}*: {value}")
        
        lines.extend([
            "",
            f"_Churn Radar • {notification.timestamp}_",
        ])
        
        return "\n".join(lines)
    
    async def send(self, notification: Notification) -> bool:
        """Send notification to Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": self._build_message(notification),
            "parse_mode": "Markdown",
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=10.0)
                
                if response.status_code == 200:
                    logger.info(f"Telegram notification sent: {notification.title}")
                    return True
                else:
                    logger.error(f"Telegram error: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")
            return False


class ConsoleChannel(NotificationChannel):
    """Console/log notification channel (for testing)."""
    
    async def send(self, notification: Notification) -> bool:
        """Log notification to console."""
        level_map = {
            NotificationLevel.INFO: logger.info,
            NotificationLevel.SUCCESS: logger.info,
            NotificationLevel.WARNING: logger.warning,
            NotificationLevel.ERROR: logger.error,
        }
        
        log_fn = level_map.get(notification.level, logger.info)
        log_fn(f"[{notification.level.value.upper()}] {notification.title}: {notification.message}")
        
        if notification.metadata:
            logger.info(f"Metadata: {notification.metadata}")
        
        return True


class NotificationManager:
    """Manages multiple notification channels."""
    
    def __init__(self):
        self.channels: List[NotificationChannel] = []
    
    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.channels.append(channel)
    
    async def send(self, notification: Notification) -> Dict[str, bool]:
        """Send notification to all channels.
        
        Returns dict mapping channel name to success status.
        """
        results = {}
        
        for channel in self.channels:
            channel_name = type(channel).__name__
            try:
                success = await channel.send(notification)
                results[channel_name] = success
            except Exception as e:
                logger.error(f"Channel {channel_name} failed: {e}")
                results[channel_name] = False
        
        return results
    
    async def notify_training_complete(
        self,
        model_name: str,
        metrics: Dict[str, float],
        training_time_sec: float,
    ) -> None:
        """Send training completion notification."""
        notification = Notification(
            title="🎓 Model Training Complete",
            message=f"Model `{model_name}` has finished training.",
            level=NotificationLevel.SUCCESS,
            metadata={
                "Model": model_name,
                "ROC-AUC": f"{metrics.get('roc_auc', 0):.4f}",
                "F1 Score": f"{metrics.get('f1', 0):.4f}",
                "Training Time": f"{training_time_sec:.1f}s",
            },
        )
        await self.send(notification)
    
    async def notify_model_promoted(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> None:
        """Send model promotion notification."""
        notification = Notification(
            title="🚀 Model Promoted",
            message=f"Model `{model_name}` v{version} promoted to {stage}.",
            level=NotificationLevel.SUCCESS,
            metadata={
                "Model": model_name,
                "Version": version,
                "Stage": stage,
            },
        )
        await self.send(notification)
    
    async def notify_pipeline_failed(
        self,
        pipeline_name: str,
        error: str,
        task: Optional[str] = None,
    ) -> None:
        """Send pipeline failure notification."""
        notification = Notification(
            title="❌ Pipeline Failed",
            message=f"Pipeline `{pipeline_name}` encountered an error.",
            level=NotificationLevel.ERROR,
            metadata={
                "Pipeline": pipeline_name,
                "Task": task or "Unknown",
                "Error": error[:200],
            },
        )
        await self.send(notification)
    
    async def notify_drift_detected(
        self,
        features: List[str],
        severity: str,
    ) -> None:
        """Send data drift notification."""
        notification = Notification(
            title="⚠️ Data Drift Detected",
            message=f"Significant drift detected in {len(features)} features.",
            level=NotificationLevel.WARNING,
            metadata={
                "Features": ", ".join(features[:5]) + ("..." if len(features) > 5 else ""),
                "Severity": severity,
            },
        )
        await self.send(notification)


# Global notification manager
_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get global notification manager."""
    global _manager
    if _manager is None:
        _manager = NotificationManager()
        
        # Add console channel by default
        _manager.add_channel(ConsoleChannel())
        
        # Add Slack if configured
        if os.getenv("SLACK_WEBHOOK_URL"):
            _manager.add_channel(SlackChannel())
        
        # Add Telegram if configured
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            _manager.add_channel(TelegramChannel())
    
    return _manager


# Convenience functions for sync code
def notify_sync(notification: Notification) -> None:
    """Synchronous notification helper."""
    import asyncio
    
    manager = get_notification_manager()
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(manager.send(notification))
