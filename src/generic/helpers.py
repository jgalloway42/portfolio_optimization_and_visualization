"""
Data Science Helper Functions Module

This module provides a collection of utility functions for common data science tasks
including data manipulation, visualization, file operations, and logging.

Dependencies:
    - numpy: Numerical computing
    - pandas: Data manipulation and analysis
    - matplotlib: Static plotting
    - seaborn: Statistical data visualization
    - plotly: Interactive plotting
    - itertools: Iterator functions
    - os: Operating system interface
    - datetime: Date and time handling
    - joblib: Efficient serialization of Python objects
    - logging: Flexible event logging system
    - pathlib: Modern path handling
    - typing: Type hints

Author: [Your Name]
Created: [Date]
Last Modified: [Date]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from itertools import chain
import os
from datetime import datetime
import joblib
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Union

# =============================================================================
# DATA CATALOG AND FILE DISCOVERY FUNCTIONS
# =============================================================================

class DataCatalog:
    """
    Enhanced file catalog specifically designed for data science projects.
    Provides better organization, metadata, and integration with pandas.
    """
    
    def __init__(self, root_directory: str):
        """
        Initialize the DataCatalog with a root directory.
        
        Args:
            root_directory (str): Path to the project root directory
        """
        self.root = Path(root_directory)
        self.catalog = pd.DataFrame()
        
    def scan_directory(self, 
                      subdirs: Optional[List[str]] = None,
                      file_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scan directories and create a comprehensive file catalog.
        
        Args:
            subdirs (list, optional): Specific subdirectories to scan. 
                Defaults to ['raw', 'processed', 'interim', 'external'].
            file_types (list, optional): File extensions to include. 
                Defaults to common data formats.
        
        Returns:
            pandas.DataFrame: DataFrame with file metadata including basename,
                filename, directory, path, extension, size, and modification date.
        """
        subdirs = subdirs or ['raw', 'processed', 'interim', 'external']
        file_types = file_types or ['.csv', '.xlsx', '.json', '.pkl', '.parquet', '.h5', '.xls']
        
        catalog_data = []
        
        for subdir in subdirs:
            subdir_path = self.root / 'data' / subdir
            if not subdir_path.exists():
                continue
                
            for file_path in subdir_path.rglob('*'):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in file_types and
                    not file_path.name.startswith('.') and
                    file_path.name != '.gitkeep'):
                    
                    # Get file metadata
                    stat = file_path.stat()
                    
                    catalog_data.append({
                        'basename': file_path.stem,
                        'filename': file_path.name,
                        'directory': subdir,
                        'path': str(file_path),
                        'extension': file_path.suffix,
                        'size_mb': round(stat.st_size / (1024 * 1024), 3),
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'relative_path': str(file_path.relative_to(self.root))
                    })
        
        self.catalog = pd.DataFrame(catalog_data)
        return self.catalog
    
    def find_files(self, pattern: str) -> pd.DataFrame:
        """
        Find files matching a pattern in basename or filename.
        
        Args:
            pattern (str): Search pattern to match against file names
            
        Returns:
            pandas.DataFrame: Filtered DataFrame with matching files
        """
        if self.catalog.empty:
            self.scan_directory()
        
        mask = (self.catalog['basename'].str.contains(pattern, case=False, na=False) |
                self.catalog['filename'].str.contains(pattern, case=False, na=False))
        return self.catalog[mask]
    
    def get_path(self, basename: str) -> Optional[str]:
        """
        Get the full path for a file by its basename.
        
        Args:
            basename (str): File basename (without extension)
            
        Returns:
            str or None: Full file path if unique match found, None otherwise
        """
        matches = self.catalog[self.catalog['basename'] == basename]
        if matches.empty:
            print(f"No file found with basename: '{basename}'")
            return None
        elif len(matches) > 1:
            print(f"Multiple files found for '{basename}':")
            print(matches[['filename', 'directory', 'relative_path']])
            return None
        else:
            return matches.iloc[0]['path']
    
    def summary(self) -> pd.DataFrame:
        """
        Get summary statistics by directory and file type.
        
        Returns:
            pandas.DataFrame: Grouped summary with file counts, sizes, and dates
        """
        if self.catalog.empty:
            self.scan_directory()
            
        if self.catalog.empty:
            return pd.DataFrame()
            
        return (self.catalog.groupby(['directory', 'extension'])
                .agg({
                    'filename': 'count',
                    'size_mb': ['sum', 'mean'],
                    'modified': 'max'
                })
                .round(3))
    
    def load_file(self, basename: str, **kwargs):
        """
        Load a file by basename using appropriate pandas method.
        
        Args:
            basename (str): File basename to load
            **kwargs: Additional arguments passed to pandas loader
            
        Returns:
            pandas.DataFrame or object: Loaded data
            
        Raises:
            FileNotFoundError: If no file found with the given basename
            ValueError: If file type is not supported
        """
        file_path = self.get_path(basename)
        if not file_path:
            raise FileNotFoundError(f"No file found with basename: {basename}")
        
        path = Path(file_path)
        
        if path.suffix.lower() == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, **kwargs)
        elif path.suffix.lower() == '.json':
            return pd.read_json(file_path, **kwargs)
        elif path.suffix.lower() == '.pkl':
            return pd.read_pickle(file_path, **kwargs)
        elif path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        elif path.suffix.lower() == '.h5':
            # For HDF5, user needs to specify the key
            if 'key' not in kwargs:
                raise ValueError("HDF5 files require a 'key' parameter")
            return pd.read_hdf(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")


def create_data_catalog(root_dir: str) -> DataCatalog:
    """
    Convenience function to create and populate a data catalog.
    
    Args:
        root_dir (str): Path to the project root directory
        
    Returns:
        DataCatalog: Initialized and populated data catalog
        
    Example:
        >>> catalog = create_data_catalog('.')
        >>> catalog.summary()  # Show overview
        >>> df = catalog.load_file('sales_data')  # Load file by basename
        >>> catalog.find_files('sales')  # Search for files
    """
    catalog = DataCatalog(root_dir)
    catalog.scan_directory()
    return catalog


# =============================================================================
# GENERAL HELPER FUNCTIONS
# =============================================================================
def walk_directory(dir_to_walk):
    """
    Recursively walk through a directory and catalog all files with their full paths.
    
    This function traverses a directory tree and creates a dictionary mapping
    file basenames (without extensions) to their full file paths. It's particularly
    useful for data science projects to quickly discover and access datasets
    stored in various subdirectories.
    
    Args:
        dir_to_walk (str): The root directory path to traverse. Can be absolute 
            or relative path. The function will recursively explore all 
            subdirectories within this path.
    
    Returns:
        dict: A dictionary where:
            - Keys are filenames without extensions (e.g., 'dataset1' from 'dataset1.csv')
            - Values are full file paths (e.g., '/path/to/data/raw/dataset1.csv')
            
    Example:
        >>> # Directory structure:
        >>> # data/
        >>> #   raw/
        >>> #     sales_data.csv
        >>> #     customer_info.xlsx
        >>> #   processed/
        >>> #     cleaned_sales.pkl
        >>> 
        >>> files = walk_directory('data/')
        sales_data /path/to/data/raw/sales_data.csv
        customer_info /path/to/data/raw/customer_info.xlsx
        cleaned_sales /path/to/data/processed/cleaned_sales.pkl
        >>> 
        >>> # Access files using the returned dictionary
        >>> sales_path = files['sales_data']
        >>> df = pd.read_csv(sales_path)
    
    Side Effects:
        - Prints each discovered file in the format: "basename full_path"
        - Output helps users see what files are available at a glance
    
    Notes:
        - Files named '.gitkeep' are automatically ignored (common Git placeholder files)
        - If multiple files have the same basename but different extensions,
          only the last one encountered will be retained in the dictionary
        - The function handles nested directory structures of any depth
        - Empty directories are traversed but produce no output
        
    Raises:
        OSError: If the specified directory doesn't exist or isn't accessible
        PermissionError: If the function lacks permissions to read certain directories
        
    Best Practices:
        - Use with data science directory structures (raw/, processed/, interim/)
        - Store the returned dictionary to access files programmatically later
        - Combine with quick_load() or similar functions for streamlined data access
    """
    # Initialize dictionary to store filename -> filepath mappings
    files_dict = {}   
    
    # Define files to ignore during directory traversal
    # .gitkeep files are commonly used in Git to preserve empty directories
    ignore = ['.gitkeep']
    
    # Recursively walk through the directory tree
    # os.walk() returns (current_directory, subdirectories, filenames) for each level
    for dirname, _, filenames in os.walk(dir_to_walk):     
        
        # Process each file in the current directory
        for filename in filenames:                      
            
            # Skip files that are in the ignore list
            if filename in ignore:                              
                continue                      
            
            # Extract the basename (filename without extension) for use as dictionary key
            # e.g., 'dataset.csv' -> 'dataset'
            basename = filename.split('.')[0]
            
            # Create the full file path by joining directory and filename
            full_path = os.path.join(dirname, filename)
            
            # Store the mapping: basename -> full_path
            files_dict[basename] = full_path
            
            # Print discovery information for user feedback
            # Format: "basename full_path"
            print(basename, full_path)
    
    # Return the complete dictionary of discovered files
    return files_dict

def search_columns(search_str, df):
    """
    Search for columns in a DataFrame that contain a specific substring.
    
    Performs a case-insensitive search through all column names in a DataFrame
    and returns a list of column names that contain the search string.
    
    Args:
        search_str (str): The substring to search for in column names.
        df (pandas.DataFrame): The DataFrame whose columns will be searched.
    
    Returns:
        list: A list of column names that contain the search string.
              Returns empty list if no matches found.
    
    Example:
        >>> df = pd.DataFrame({'Sales_2021': [1, 2], 'Revenue_2021': [3, 4], 'Costs': [5, 6]})
        >>> search_columns('2021', df)
        ['Sales_2021', 'Revenue_2021']
        
        >>> search_columns('sales', df)  # Case insensitive
        ['Sales_2021']
    
    Note:
        The search is case-insensitive. Both the search string and column names
        are converted to uppercase for comparison.
    """
    return list(filter(
        lambda x: search_str.upper() in x.upper(), 
        [y for y in df.columns]
    ))


def filter_list(search_str, searchable_list):
    """
    Filter a list of strings based on a substring match.
    
    Performs a case-insensitive search through a list of strings and returns
    only those items that contain the specified search string.
    
    Args:
        search_str (str): The substring to search for in list items.
        searchable_list (list): List of strings to search through.
    
    Returns:
        list: A filtered list containing only items that match the search criteria.
              Returns empty list if no matches found.
    
    Example:
        >>> items = ['apple_pie', 'banana_bread', 'apple_tart', 'orange_juice']
        >>> filter_list('apple', items)
        ['apple_pie', 'apple_tart']
        
        >>> filter_list('BREAD', items)  # Case insensitive
        ['banana_bread']
    
    Note:
        The search is case-insensitive. Both the search string and list items
        are converted to uppercase for comparison.
    """
    return list(filter(
        lambda x: search_str.upper() in x.upper(), 
        [y for y in searchable_list]
    ))


def flatten_list(list_of_lists):
    """
    Flatten a nested list structure into a single-level list.
    
    Takes a list containing sublists and returns a single flat list with all
    elements from the nested structure. Uses itertools.chain for efficiency.
    
    Args:
        list_of_lists (list): A list containing sublists or iterables.
    
    Returns:
        list: A flattened list containing all elements from the nested structure.
    
    Example:
        >>> nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        >>> flatten_list(nested)
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        >>> mixed = [['a', 'b'], ['c'], ['d', 'e']]
        >>> flatten_list(mixed)
        ['a', 'b', 'c', 'd', 'e']
    
    Note:
        This function assumes all elements in the input list are iterable.
        It will raise an error if any element is not iterable.
    """
    return list(chain(*list_of_lists))


# =============================================================================
# PLOTTING AND VISUALIZATION FUNCTIONS
# =============================================================================

def plot_df(data, cols_list, save_to_path=None, figsize=(15, 20), 
            linestyle='none', marker=','):
    """
    Create subplot visualizations for specified DataFrame columns.
    
    Generates individual subplots for each specified column in the DataFrame.
    Supports caching by reading from file if plot already exists, or creating
    and optionally saving new plots.
    
    Args:
        data (pandas.DataFrame): The DataFrame containing the data to plot.
        cols_list (list): List of column names to plot as separate subplots.
        save_to_path (str, optional): File path to save the plot. If None, 
            plot is displayed but not saved. Defaults to None.
        figsize (tuple, optional): Figure size as (width, height) in inches. 
            Defaults to (15, 20).
        linestyle (str, optional): Line style for the plot. Defaults to 'none'.
        marker (str, optional): Marker style for data points. Defaults to ','.
    
    Returns:
        str: Status message indicating whether plot was read from file, 
             saved to file, or displayed without saving.
    
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> plot_df(df, ['A', 'B'], save_to_path='my_plot.png')
        'Saved to File: my_plot.png'
        
        >>> plot_df(df, ['A', 'B'])  # Display only
        'graph not saved to file...'
    
    Note:
        - If save_to_path exists, the function loads and displays the existing image
        - The function uses matplotlib's subplot functionality
        - Large figsize may be needed for multiple subplots to be readable
    """
    if save_to_path:
        # Check if plot already exists and load it
        if os.path.exists(save_to_path):
            plt.imshow(plt.imread(save_to_path), aspect='auto', interpolation='nearest')
            return f'Read From File: {save_to_path}'
        else:
            # Create new plot and save it
            _ = data[cols_list].plot(
                subplots=True, 
                linestyle=linestyle,
                marker=marker, 
                figsize=figsize
            )
            plt.savefig(save_to_path)
            return f'Saved to File: {save_to_path}'
    else:
        # Create plot without saving
        _ = data[cols_list].plot(
            subplots=True, 
            linestyle=linestyle, 
            marker=',', 
            figsize=(15, 30)
        )
        return 'graph not saved to file...'


def plotly_graph(df, cols_to_graph, left_legend=False, save_to_path=None):
    """
    Create interactive line plots using Plotly Express.
    
    Generates an interactive line chart with multiple series using the DataFrame
    index as the x-axis and specified columns as y-values. Supports legend
    positioning and HTML export.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        cols_to_graph (list or str): Column name(s) to plot as line series.
        left_legend (bool, optional): If True, positions legend on the left side.
            Defaults to False.
        save_to_path (str, optional): File path to save the plot as HTML. 
            If None, plot is displayed interactively. Defaults to None.
    
    Returns:
        None: Displays the plot or saves it to file.
    
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> plotly_graph(df, ['A', 'B'], left_legend=True)
        # Displays interactive plot with legend on left
        
        >>> plotly_graph(df, 'A', save_to_path='interactive_plot.html')
        # Saves plot as HTML file
    
    Note:
        - Uses DataFrame index as x-axis values
        - Supports both single column (str) and multiple columns (list)
        - HTML files can be opened in web browsers for interactivity
        - Legend positioning uses Plotly's layout update functionality
    """
    # Create the line plot using DataFrame index as x-axis
    f = px.line(data_frame=df, x=df.index, y=cols_to_graph)
    
    # Position legend on the left if requested
    if left_legend:
        f.update_layout(legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="left", 
            x=0.01
        ))
    
    # Save to file or display
    if save_to_path:
        print(f'...saving graph to {save_to_path}')
        f.write_html(save_to_path)
    else:
        f.show()


# =============================================================================
# FILE OPERATIONS AND PERSISTENCE FUNCTIONS
# =============================================================================

def save_joblib(object, folder_path, file_name, add_timestamp=False):
    """
    Save Python objects to disk using joblib serialization.
    
    Serializes and saves Python objects (models, data structures, etc.) to disk
    using joblib's efficient compression. Optionally adds timestamp to filename
    to prevent overwrites.
    
    Args:
        object: The Python object to be saved (can be any serializable object).
        folder_path (str): Directory path where the file will be saved.
        file_name (str): Base name for the saved file (including extension).
        add_timestamp (bool, optional): If True, adds current timestamp to filename.
            Defaults to False.
    
    Returns:
        None: Prints confirmation message with full file path.
    
    Example:
        >>> model = {'type': 'regression', 'accuracy': 0.95}
        >>> save_joblib(model, './models', 'my_model.pkl')
        File Saved: ./models/my_model.pkl
        
        >>> save_joblib(model, './models', 'model.pkl', add_timestamp=True)
        File Saved: ./models/model_2023_12_15_14_30_45.pkl
    
    Note:
        - Joblib is particularly efficient for NumPy arrays and scikit-learn models
        - Timestamp format: YYYY_MM_DD_HH_MM_SS
        - Creates the full file path by joining folder_path and file_name
        - Does not create directories; folder_path must exist
    """
    if add_timestamp:
        # Generate timestamp and modify filename
        current_timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        root, ext = os.path.splitext(file_name)
        file_name = root + '_' + current_timestamp + ext  # ext retains the .xxxx including the .
    
    # Save the object using joblib
    full_path = os.path.join(folder_path, file_name)
    joblib.dump(object, full_path)
    print(f'File Saved: {full_path}')


def get_logger(folder_path, file_name, logging_level=logging.DEBUG, add_timestamp=True):
    """
    Create and configure a logger for file-based logging.
    
    Sets up a Python logger with file handler for persistent logging during
    data science workflows. Configures formatting and logging levels, with
    optional timestamp addition to log filenames.
    
    Args:
        folder_path (str): Directory path where log files will be saved.
        file_name (str): Base name for the log file (including extension).
        logging_level (int, optional): Logging level (e.g., logging.DEBUG, 
            logging.INFO). Defaults to logging.DEBUG.
        add_timestamp (bool, optional): If True, adds current timestamp to filename.
            Defaults to True.
    
    Returns:
        logging.Logger: Configured logger object ready for use.
    
    Example:
        >>> logger = get_logger('./logs', 'analysis.log')
        Log Started: ./logs/analysis_2023_12_15_14_30_45.log
        >>> logger.info('Starting data analysis')
        >>> logger.error('Failed to load dataset')
    
    Available Logging Levels:
        - logging.DEBUG: Detailed information for debugging
        - logging.INFO: General information about program execution
        - logging.WARNING: Indication of potential problems
        - logging.ERROR: Error occurred but program continues
        - logging.CRITICAL: Serious error, program may not continue
    
    Note:
        - Logger name is set to 'Jupyter Notebook'
        - Log format includes timestamp, logger name, level, and message
        - Multiple calls create handlers on the same logger (may cause duplicates)
        - Consider using logging.getLogger(__name__) for module-specific loggers
    """
    if add_timestamp:
        # Generate timestamp and modify filename
        current_timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        root, ext = os.path.splitext(file_name)
        file_name = root + '_' + current_timestamp + ext  # ext retains the .xxxx including the .
    
    # Create and configure logger
    logger = logging.getLogger('Jupyter Notebook')
    logger.setLevel(logging_level)
    
    # Create file handler
    full_path = os.path.join(folder_path, file_name)
    handler = logging.FileHandler(full_path)
    handler.setLevel(logging_level)
    
    # Create formatter and add it to handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    print(f'Log Started: {full_path}')
    return logger