from abc import ABC, abstractmethod

class Tracker(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Abstract property for tracker's name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Abstract property for tracker's description"""
        pass
    
    @abstractmethod
    def get_custom_inputs(self):
        """Abstract method to get inputs"""
        pass
    
    @abstractmethod
    def track(self, video_path, keypoints, source_frame_percentage, custom_inputs) -> str:
        """Abstract method to perform tracking"""
        pass


