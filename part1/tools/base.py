from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

class Tool(ABC):
    """Base class for all tools in the system"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"tool.{name}")
    
    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        """Execute the tool with given input"""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input before execution"""
        return True
    
    async def __call__(self, input_data: Any) -> Any:
        """Make tool callable"""
        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input for tool {self.name}")
        
        self.logger.info(f"Executing {self.name}")
        try:
            result = await self.execute(input_data)
            self.logger.info(f"{self.name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} failed: {str(e)}")
            raise