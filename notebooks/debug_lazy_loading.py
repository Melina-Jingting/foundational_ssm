#!/usr/bin/env python3
"""
Debug script to trace where LazyInterval objects are coming from when lazy=False.

This script will help identify the exact point in the data loading pipeline where
lazy objects are being created despite setting lazy=False.
"""

import sys
import traceback
import inspect
from typing import Any, Dict, List, Optional
import h5py
import numpy as np

# Add the temporaldata path to sys.path if needed
sys.path.insert(0, '/cs/student/projects1/ml/2024/mlaimon/temporaldata')

from temporaldata import Data, Interval, LazyInterval, ArrayDict, LazyArrayDict
from temporaldata import IrregularTimeSeries, LazyIrregularTimeSeries
from temporaldata import RegularTimeSeries, LazyRegularTimeSeries

# Import the dataset classes
from foundational_ssm.data_utils.dataset import TorchBrainDataset
from foundational_ssm.data_utils.loaders import get_dataset_config
from foundational_ssm.constants import DATA_ROOT
from omegaconf import OmegaConf

class LazyObjectTracker:
    """Tracks where lazy objects are being created."""
    
    def __init__(self):
        self.lazy_objects_created = []
        self.creation_stack_traces = []
        
    def track_object_creation(self, obj, context=""):
        """Track when a lazy object is created."""
        if self._is_lazy_object(obj):
            stack_trace = traceback.format_stack()
            self.lazy_objects_created.append({
                'object': obj,
                'object_type': type(obj).__name__,
                'context': context,
                'stack_trace': stack_trace,
                'creation_location': self._get_creation_location(stack_trace)
            })
            print(f"🚨 LAZY OBJECT CREATED: {type(obj).__name__} in context: {context}")
            print(f"   Location: {self._get_creation_location(stack_trace)}")
    
    def _is_lazy_object(self, obj):
        """Check if an object is a lazy variant."""
        lazy_classes = [
            LazyArrayDict, LazyInterval, LazyIrregularTimeSeries, 
            LazyRegularTimeSeries
        ]
        return any(isinstance(obj, cls) for cls in lazy_classes)
    
    def _get_creation_location(self, stack_trace):
        """Extract the most relevant location from stack trace."""
        for line in stack_trace:
            if 'temporaldata' in line and ('from_hdf5' in line or '__init__' in line):
                return line.strip()
        return stack_trace[-2] if len(stack_trace) > 1 else stack_trace[-1]
    
    def print_summary(self):
        """Print a summary of all lazy objects created."""
        print("\n" + "="*80)
        print("LAZY OBJECTS CREATION SUMMARY")
        print("="*80)
        
        if not self.lazy_objects_created:
            print("✅ No lazy objects were created!")
            return
        
        print(f"🚨 Found {len(self.lazy_objects_created)} lazy objects created:")
        print()
        
        for i, info in enumerate(self.lazy_objects_created):
            print(f"{i+1}. {info['object_type']} - {info['context']}")
            print(f"   Location: {info['creation_location']}")
            print()

# Global tracker
tracker = LazyObjectTracker()

def debug_data_from_hdf5(file, lazy=True, context=""):
    """Debug version of Data.from_hdf5 that tracks lazy object creation."""
    print(f"🔍 Loading Data from HDF5 with lazy={lazy}, context={context}")
    
    # Check that the file is read-only
    if isinstance(file, h5py.File):
        assert file.mode == "r", "File must be opened in read-only mode."

    data = {}
    for key, value in file.items():
        if isinstance(value, h5py.Group):
            class_name = value.attrs["object"]
            print(f"  📁 Processing group '{key}' with class_name='{class_name}'")
            
            if lazy and class_name != "Data":
                group_cls = globals()[f"Lazy{class_name}"]
                print(f"    -> Using Lazy class: {group_cls.__name__}")
            else:
                group_cls = globals()[class_name]
                print(f"    -> Using Regular class: {group_cls.__name__}")
            
            obj = group_cls.from_hdf5(value)
            tracker.track_object_creation(obj, f"group '{key}' (class_name='{class_name}')")
            data[key] = obj
        else:
            # if array, it will be loaded no matter what, always prefer ArrayDict
            data[key] = value[:]

    for key, value in file.attrs.items():
        if key == "object" or key == "absolute_start":
            continue
        data[key] = value

    obj = Data(**data)

    # restore the absolute start time
    obj._absolute_start = file.attrs["absolute_start"]

    # Load domain if it exists
    if "domain" in file:
        domain_group = file["domain"]
        domain_class_name = domain_group.attrs["object"]
        print(f"  📁 Processing domain with class_name='{domain_class_name}'")
        
        if lazy and domain_class_name != "Data":
            domain_cls = globals()[f"Lazy{domain_class_name}"]
            print(f"    -> Using Lazy domain class: {domain_cls.__name__}")
        else:
            domain_cls = globals()[domain_class_name]
            print(f"    -> Using Regular domain class: {domain_cls.__name__}")
        
        domain_obj = domain_cls.from_hdf5(domain_group)
        tracker.track_object_creation(domain_obj, f"domain (class_name='{domain_class_name}')")
        obj._domain = domain_obj

    return obj

def debug_dataset_creation():
    """Debug the dataset creation process."""
    print("🔍 Starting dataset creation debug...")
    
    config_path = "/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/configs/pretrain.yaml"
    cfg = OmegaConf.load(config_path)
    
    print(f"📋 Config lazy setting: {cfg.dataloader.lazy}")
    print(f"📋 Config keep_files_open: {cfg.dataloader.keep_files_open}")
    
    # Create dataset with lazy=False
    train_dataset = TorchBrainDataset(
        root="../"+DATA_ROOT,
        config=get_dataset_config(**cfg.train_dataset),
        lazy=False,  # Force lazy=False for debugging
        keep_files_open=cfg.dataloader.keep_files_open,
    )
    
    print("✅ Dataset created successfully")
    return train_dataset

def debug_data_loading(train_dataset):
    """Debug the data loading process."""
    print("\n🔍 Testing data loading...")
    
    # Get a sample recording ID
    recording_ids = list(train_dataset.recording_dict.keys())
    if not recording_ids:
        print("❌ No recording IDs found!")
        return
    
    test_recording_id = recording_ids[0]
    print(f"📁 Testing with recording ID: {test_recording_id}")
    
    # Test _get_data_object
    print("\n🔍 Testing _get_data_object...")
    try:
        data_obj = train_dataset._get_data_object(test_recording_id)
        print(f"✅ Data object created: {type(data_obj).__name__}")
        
        # Check if data_obj contains any lazy objects
        print("\n🔍 Checking for lazy objects in data_obj...")
        check_object_for_lazy_objects(data_obj, "data_obj")
        
    except Exception as e:
        print(f"❌ Error in _get_data_object: {e}")
        traceback.print_exc()
    
    # Test get method
    print("\n🔍 Testing get method...")
    try:
        sample = train_dataset.get(test_recording_id, 0.0, 1.0)
        print(f"✅ Sample created: {type(sample).__name__}")
        
        # Check if sample contains any lazy objects
        print("\n🔍 Checking for lazy objects in sample...")
        check_object_for_lazy_objects(sample, "sample")
        
    except Exception as e:
        print(f"❌ Error in get method: {e}")
        traceback.print_exc()

def check_object_for_lazy_objects(obj, context=""):
    """Recursively check an object for lazy objects."""
    if tracker._is_lazy_object(obj):
        tracker.track_object_creation(obj, context)
        return
    
    # Check attributes
    if hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            if attr_name.startswith('_'):
                continue  # Skip private attributes
            check_object_for_lazy_objects(attr_value, f"{context}.{attr_name}")
    
    # Check if it's a container
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            check_object_for_lazy_objects(item, f"{context}[{i}]")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            check_object_for_lazy_objects(value, f"{context}[{key}]")

def debug_hdf5_file_structure(file_path):
    """Debug the structure of an HDF5 file."""
    print(f"\n🔍 Analyzing HDF5 file structure: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("📁 File structure:")
            print_hdf5_structure(f, indent=2)
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")

def print_hdf5_structure(group, indent=0):
    """Print the structure of an HDF5 group."""
    prefix = "  " * indent
    
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            print(f"{prefix}📁 {key}/ (Group)")
            if 'object' in value.attrs:
                print(f"{prefix}   └─ object: {value.attrs['object']}")
            print_hdf5_structure(value, indent + 1)
        else:
            print(f"{prefix}📄 {key} (Dataset: {value.shape}, {value.dtype})")
    
    # Print attributes
    if group.attrs:
        print(f"{prefix}📋 Attributes:")
        for key, value in group.attrs.items():
            print(f"{prefix}   └─ {key}: {value}")

def main():
    """Main debug function."""
    print("🚀 Starting LazyInterval Debug Script")
    print("="*80)
    
    try:
        # Debug dataset creation
        train_dataset = debug_dataset_creation()
        
        # Debug data loading
        debug_data_loading(train_dataset)
        
        # Print summary
        tracker.print_summary()
        
        # Debug HDF5 file structure if we have a recording
        if hasattr(train_dataset, 'recording_dict') and train_dataset.recording_dict:
            first_recording = list(train_dataset.recording_dict.keys())[0]
            file_path = train_dataset.recording_dict[first_recording]["filename"]
            debug_hdf5_file_structure(file_path)
        
    except Exception as e:
        print(f"❌ Error in main debug: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 