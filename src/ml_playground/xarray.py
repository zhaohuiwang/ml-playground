# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:28:15 2020
http://xarray.pydata.org/en/stable/index.html
@author: Family_Optiplex9


Xarray provides two main data structures:  DataArray and Dataset

The `DataArray` class attaches metadata (name, dimension or axes names, coordinate labels and attributes)to multi-dimensional arrays (numpy.ndarray) while `Dataset` combines multiple DataArrays. Dataset objects collect multiple data variables, each with possibly different dimensions.

class xarray.DataArray(data=<NA>, coords=None, dims=None, name=None, attrs=None, indexes=None, fastpath=False)

    data: array_like - Values for this array. 
    attribute (attrs): dict key-value pairs or sequence - metadata, Attributes to assign to the new instance. ignored by operations.

    Dimension (dims): sequence or hashable - Name(s) of the data dimension(s). If this argument is omitted, dimension names are taken from coords (if possible) and otherwise default to ['dim_0', ... 'dim_n']. Must match the data dimension

    (Dimension) coordinates (coords): dict key-value pairs - describe data.Coordinates (tick labels) to use for indexing along each dimension. coords takes a dictionary that maps the name of a dimension to one of
    1. another DataArray object
    2. a tuple of the form (dims, data, attrs) where attrs is optional. This is roughly equivalent to creating a new DataArray object with DataArray(dims=dims, data=data, attrs=attrs)
    3. a numpy array (or anything that can be coerced to one using numpy.array).
    
    Non-dimension coordinates: a coordinate variable that does not match a dimension name. (not marked with the * )

    A dimension may not have coordinate, if one or more such dimension exist, they will be revealed by (whey you call the xarray object) "Dimensions without coordinates: XXXXX", and you may assign additiona coordinates, e.g.
    dataarray.coords["dim_name"] = np.arange(75, 14.9, -2.5) similarly add additional attribute
    dataarray.attrs["attribute"] = "hello"

    name: string - Name of this array (optional).

class xarray.Dataset(data_vars=None, coords=None, attrs=None)

    data_vars: dict-like -
    coords: Coordinates or dict-like - 
    attrs: dict-like - Global attributes to save on this dataset.
    # Create a Dataset with two Data variables: air and air2
    ds = xr.Dataset({"air": da, "air2": da2})
    # Assign a new Data variable directly
    ds["air3"] = da


To display data, dimension, coordinates or attribute   
    <DataArray>.<data|dims|coords|attrs>
    <Dataset>.<data|dims|coords|attrs>  
    <Dataset>.<Data_variables>.<data|dims|coords|attrs>  # or specific variable

    
# conver a series to xarray
series.to_xarray()
# convert DataArray objects to pandas.Series
ds.air.to_series()
# convert DataArray or Dataset objects to a pandas.DataFrame
ds.air.to_dataframe()


Xarray: Extracting data or “indexing” : .sel (label-based), .isel (position-based indexing)

The recommended way to store xarray data structures is NetCDF, which is
a binary file format for self-described datasets that
originated in the geosciences.
Xarray is based on the netCDF data model, so netCDF files on disk directly correspond to Dataset objects.

Xarray reads and writes to NetCDF files using the open_dataset / open_dataarray functions and the to_netcdf method.

the dimension of the data must match the dims values

A DataArray can have more coordinates than dimensions because a single dimension can be labeled by multiple coordinate arrays. However, only one coordinate array can be a assigned as a particular dimension's dimension coordinate array.


https://tutorial.xarray.dev/intro.html

"""
series = pd.Series(np.ones((10,)), index=list("abcdefghij"))
arr = series.to_xarray()    # conver a serires to xarray
arr.to_pandas()             # convert xarray to pandas
arr.to_dataframe()          # convert xarray to dataframe


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

da = xr.DataArray(
    data=np.ones((3, 4, 2)),
    dims=("x", "y", "z"), # a tuple or list or string if only one dimension, must match the dimension of the data 
    name="a",
    coords={
        "x": ["a", "b", "c"],
        "y": np.arange(4),
        "u": ("x", np.arange(3), {"attr1": 0}),
    },
    attrs={"attr": "value"},
)
type(da.data)

'''
name= : the name of DataArray, optional 
dims= : a tuple/list or single string (if 1-D) for dimension names
coords= : a dict-like container of arrays (coordinates) that label each point
    if it is a dimension coordinate, it will be marked a with a *
attrs= : a dict-like container, arbitrary metadata/attributes to the DataArray

dimentional coordinate (dimention labels):
    only possible for indexing operations (sel, reindex, etc)
    have to be one dimentional
    if it is a dimension coordinate, it will be marked a with a *
normal coordinate:
    can have arbitrary dimensions.
'''
# Once we have created the DataArray, we can look at its data:
da.data; type(da.data)
da.dims
da.coords
da.attrs

# xarray has two representation types: html and string
with xr.set_options(display_style="html"):
    display(da)
    
with xr.set_options(display_style="text"):
    display(da)

# create dataset
times = pd.date_range(start='2000-01-01',freq='1H',periods=6)
ds_c = xr.Dataset({
    'SWdown': xr.DataArray(
                data   = np.random.random(6),   # enter data here
                dims   = ['time'],
                coords = {'time': times},
                attrs  = {
                    '_FillValue': -999.9,
                    'units'     : 'W/m2'
                    }
                ),
    'LWdown': xr.DataArray(
                data   = np.ones((6, 4, 2)),   # enter data here
                dims   = ['time', 's0', 'dd'],
                coords = {'time': times},
                attrs  = {
                    '_FillValue': -999.9,
                    'units'     : 'W/m2'
                    }
                )
            },
        attrs = {'example_attr': 'this is a global attribute'}
    )

# load tutorial dataset
ds = xr.tutorial.load_dataset("air_temperature")

# create additiona coordinates
array = ds.air.data
lon_values = np.arange(200, 331, 2.5)
# assing a plain numpy.ndarray to coordinate
da = xr.DataArray(data=array, dims=("time", "lat", "lon"), coords={"lon": lon_values})
# Assigning a plain numpy array is equivalent to creating a DataArray with those values and the same dimension name
da = xr.DataArray(data=array, dims=("time", "lat", "lon"), coords={"lon": xr.DataArray(lon_values, dims="lon")})
da.coords["lat"] = np.arange(75, 14.9, -2.5)


# Datasets are dict-like containers of DataArrays i.e. they are a mapping of variable name to DataArray.

ds["air"]    # dict-like 
ds.air      # pull out dataarray using dot notation
# This won’t work for variable names that clash with a built-in method name
# (like mean for example).

# What’s in a DataArray? data + (a lot of) metadata
# Named dimensions
ds.air.dims; ds.dims

# Coordinate variables or “tick labels” 
ds.air.coords; ds.coords

# extracting coordinate variables
ds.air.lon; ds.lon

# Arbitrary attributes
# a dictionary that can contain arbitrary python objects.
# Your only limitation is
# that some attributes may not be writeable to a netCDF file

ds.attrs; ds.air.attrs
# assign your own attribute
ds.air.attrs["who_is_awesome"] = "xarray"; ds.air.attrs

# Underlying data
ds.air
# Xarray structures wrap underlying simpler data structures.
# a numpy array, GPU arrays, sparse arrays, arrays with units etc.
ds.air.data


# what is the type of the underlying data
type(ds.air.data)

ds.air.isel(time=1).plot(x="lon")

ds.air.mean("time")

# Label-based indexing
# pull out data for all of 2013-May
ds.sel(time="2013-05")

# demonstrate slicing
ds.sel(time=slice("2013-05", "2013-07"))

# demonstrate "nearest" indexing
ds.sel(lon=240.2, method="nearest")

# "nearest indexing at multiple points"
ds.sel(lon=[240.125, 234], lat=[40.3, 50.3], method="nearest")

# Position-based indexing
# pull out time index 0 and lat index 0
ds.air.isel(time=0, lat=0)  #  much better than ds.air[0, 0, :]

# demonstrate slicing
ds.air.isel(lat=slice(10))

# Concepts for computation
# Broadcasting: expanding data
'''
Let's try to calculate grid cell area associated with the air temperature data.
We may want this to make a proper area-weighted domain-average for example
A very approximate formula is
Δlat X Δlon X cos(latutude)
assuming that Δlon = 111km and Δlat = 111km

The result has two dimensions because xarray realizes that
dimensions lon and lat are different so
it automatically “broadcasts” to get a 2D result.

'''

dlon = np.cos(ds.air.lat * np.pi / 180) * 111e3
dlat = xr.ones_like(ds.air.lon) * 111e3 
# Return a new object of ones with the same shape and type as
# a given dataarray or dataset.

cell_area = dlon * dlat
cell_area

dlon.shape
dlat.shape
cell_area.shape


np.arange(3) + 5
np.arange(3) + np.repeat(5, 3)

np.ones((3, 3)) + np.arange(3) 
# numpy.ones() returns a new array of given shape and type, filled with ones

np.arange(3).reshape((3,1)) + np.arange(3) 


# Alignment: putting data on the same grid
'''
When doing arithmetic operations xarray automatically “aligns” i.e.
puts the data on the same grid.
In this case cell_area and ds.air are at the same lat, lon points so
things are multiplied as you would expect.

If they can not align, ...

Tip: If you notice extra NaNs or missing points after xarray computation,
it means that your xarray coordinates were not aligned exactly.

https://xarray.pydata.org/en/stable/computation.html#automatic-alignment
'''
cell_area.shape
ds.air.isel(time=1).shape
(cell_area * ds.air.isel(time=1)).shape


# Now lets make cell_area unaligned i.e. change the coordinate labels

# make a copy of cell_area
cell_area_bad = cell_area.copy(deep=True)
# then add 1e-5 to lat
cell_area_bad["lat"] = cell_area.lat + 1e-5
cell_area_bad * ds.air.isel(time=1)
'''
only 'lon' aligns
<xarray.DataArray (lat: 0, lon: 53)>
array([], shape=(0, 53), dtype=float32)
Coordinates:
  * lat      (lat) float64 
  * lon      (lon) float32 200.0 202.5 205.0 207.5 ... 322.5 325.0 327.5 330.0
    time     datetime64[ns] 2013-01-01T06:00:00
'''
# additionally, when add 1e05 to lon
cell_area_bad["lon"] = cell_area.lon + 1e05
cell_area_bad * ds.air.isel(time=1)

'''
no coordinate variable align
<xarray.DataArray (lat: 0, lon: 0)>
array([], shape=(0, 0), dtype=float32)
Coordinates:
  * lat      (lat) float64 
  * lon      (lon) float64 
    time     datetime64[ns] 2013-01-01T06:00:00
'''

# groupby
# seasonal groups  
'''
Split-Apply-Combine Strategy

Split: Split the data into groups based on some criteria thereby
        creating a GroupBy object. 
        (We can use the column or a combination of columns to 
        split the data into groups)
Apply: Apply a function to each group independently.
        (Aggregate, Transform, or Filter the data in this step)
Combine: Combine the results into a data structure
        (Pandas Series, Pandas DataFrame)
'''
ds.groupby("time.season")

# make a seasonal mean
seasonal_mean = ds.groupby("time.season").mean()
seasonal_mean

seasonal_mean = seasonal_mean.reindex(season=["DJF", "MAM", "JJA", "SON"])
seasonal_mean

# resample
# resample to monthly frequency
ds.resample(time="M").mean()

# weighted
# weight by cell_area and take mean over (time, lon)
ds.weighted(cell_area).mean(["lon", "time"]).air.plot()


'''
Xarray has some support for visualizing 3D and 4D datasets by
presenting multiple facets (or panels or subplots) showing variations
across rows and/or columns.
'''
# facet the seasonal_mean
seasonal_mean.air.plot(col="season")

# contours
seasonal_mean.air.plot.contour(col="season", levels=20, add_colorbar=True)

# line plots too? wut
seasonal_mean.air.mean("lon").plot.line(hue="season", y="lat")

# Reading and writing to disk
# write ds to netCDF
ds.to_netcdf("my-example-dataset.nc")

# read from disk
fromdisk = xr.open_dataset("my-example-dataset.nc")
fromdisk

# check that the two are identical
ds.identical(fromdisk)


# Pandas: tabular data structures
# convert to pandas dataframe
df = ds.isel(time=slice(10)).to_dataframe()
df

# convert dataframe to xarray
df.to_xarray()



ds1 = xr.Dataset(
    data_vars={
        "a": (("x", "y"), np.random.randn(4, 2)),
        "b": (("z", "x"), np.random.randn(6, 4)),
    },
    coords={"x": np.arange(4), "y": np.arange(-2, 0), "z": np.arange(-3, 3),},
)
ds2 = xr.Dataset(
    data_vars={
        "a": (("x", "y"), np.random.randn(7, 3)),
        "b": (("z", "x"), np.random.randn(2, 7)),
    },
    coords={"x": np.arange(6, 13), "y": np.arange(3), "z": np.arange(3, 5),},
)

# write datasets
ds1.to_netcdf("ds1.nc")
ds2.to_netcdf("ds2.nc")

# write dataarray
ds1.a.to_netcdf("da1.nc")


'''
The N-dimensional array (ndarray)
https://numpy.org/doc/stable/reference/arrays.ndarray.html
'''
x = np.array([[1, 2, 3], 
              [4, 5, 6]],
             np.int32)
type(x)
x.shape
x.dtype

x[1, 2]       #  index using Python container-like syntax
x[:,1]        # slice
x[1, 2] = 9   # this also changes the corresponding element in x

 



# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:22:21 2021
https://towardsdatascience.com/basic-data-structures-of-xarray-80bab8094efa
@author: Family_Optiplex9
"""



# customary imports
import numpy as np
import pandas as pd
import xarray as xr
np.random.seed(123)
size = 4
temperature = 15 + 10 * np.random.randn(size)
lat = np.random.uniform(low=-90, high=90, size=size)
lon = np.random.uniform(low=-180, high=180, size=size)

# round to two digits after decimal point
temperature, lat , lon = np.around([temperature, lat, lon], decimals=2)


df = pd.DataFrame({"temperature":temperature, "lat":lat, "lon":lon})
df

#### Creating DataArray from Series
idx = pd.MultiIndex.from_arrays(arrays=[lat,lon], names=["lat","lon"])
s = pd.Series(data=temperature, index=idx)
s
# use from_series method
da = xr.DataArray.from_series(s)
da

####  Creating a DataArray from DataFrame
df_pv = df.pivot(index="lat", columns="lon")
# drop first level of columns as it's not necessary
df_pv = df_pv.droplevel(0, axis=1)
df_pv
# create a DataArray by providing the DataArray constructor with
# our pivoted data frame
da = xr.DataArray(data=df_pv)
da

#### Creating a DataArray using the constructor
# get pivoted data as 2-dimensional array (4,4)
temperature_2d = df_pv.values    
da = xr.DataArray(data=temperature_2d, dims=["lat", "lon"], coords=[lat,lon])
da
# alternative way to do the same thing
# xr.DataArray(data=temperature_2d, coords=[("lat",lat), ("lon",lon)])
# The important thing to notice here is that coordinate arrays must 
# be 1 dimensional and have the length of the dimension they represent. 
# We had a (4,4) shaped array of data, so we supplied the constructor with 
# two coordinate arrays. Each one is 1-dimensional and has a length of 4


#### Creating a DataArray using the constructor with projected coordinates
np.random.seed(123)
shape = (1,4)    # needs to be 2-dimensional, could be (2,2), (4,1)

# all three arrays have the same shape
temperature = 15 + 10 * np.random.randn(*shape)
lat = np.random.uniform(low=-90, high=90, size=shape)
lon = np.random.uniform(low=-180, high=180, size=shape)

# round to two digits after decimal point
temperature, lat , lon = np.around([temperature, lat, lon], decimals=2)

da = xr.DataArray(data=temperature,
                  coords={"lat": (["x","y"], lat),
                          "lon": (["x","y"], lon)},
                  dims=["x","y"])
da



# =============================================================================
# 3 dimensions
# =============================================================================

np.random.seed(123)

temperature_3d = 15 + 10 * np.random.randn(1,4,2)    # 3-dimensional
lat = np.random.uniform(low=-90, high=90, size=(1,4))
lon = np.random.uniform(low=-180, high=180, size=(1,4))

# round to two digits after decimal point
temperature_3d = np.around(temperature_3d, decimals=2)
lat , lon = np.around([lat, lon], decimals=2)

da = xr.DataArray(data=temperature_3d,
                  coords={"lat": (["x","y"], lat),
                          "lon": (["x","y"], lon), 
                          "day": ["day1","day2"]},
                  dims=["x","y","day"])
da

# make data 1-dimensional
temperature_1d = temperature_3d.flatten("F")
lat = lat.flatten()
lon = lon.flatten()
day = ["day1","day2"]

idx = pd.MultiIndex(levels=[day,lat,lon], 
                    codes=[[0]*4 + [1]*4, list(range(4))*2, list(range(4))*2], 
                    names=["day","lat","lon"])

s = pd.Series(temperature_1d, index=idx)
s

da = xr.DataArray.from_series(s)
da


# =============================================================================
# Dataset we only dealt with temperature data. Let’s add pressure data
# =============================================================================

np.random.seed(123)

# 3-dimensional temperature and pressure data
temperature_3d = 15 + 10 * np.random.randn(1,4,2)
pressure_3d = 1013 + 10 * np.random.randn(1,4,2)
lat = np.random.uniform(low=-90, high=90, size=(1,4))
lon = np.random.uniform(low=-180, high=180, size=(1,4))

# round to two digits after decimal point
temperature_3d, pressure_3d = np.around([temperature_3d, pressure_3d], decimals=2)
lat , lon = np.around([lat, lon], decimals=2)

ds = xr.Dataset(data_vars={"temperature":(["x","y","day"],temperature_3d), 
                           "pressure":(["x","y","day"],pressure_3d)}, 
                coords={"lat": (["x","y"], lat), 
                        "lon": (["x","y"], lon), 
                        "day": ["day1", "day2"]})

ds

temperature_1d, pressure_1d, lat, lon = [arr.flatten() for arr in [temperature_3d, pressure_3d, lat, lon]]    # make data 1-dimensional
day = ["day1","day2"]
idx = pd.MultiIndex(levels=[day,lat,lon], codes=[[0]*4 + [1]*4, list(range(4))*2, list(range(4))*2], names=["day","lat","lon"])

# create series
s_temperature = pd.Series(temperature_1d, index=idx)
s_pressure = pd.Series(pressure_1d, index=idx)

# create DataArrays
da_temperature = xr.DataArray.from_series(s_temperature)
da_pressure = xr.DataArray.from_series(s_pressure)

ds = xr.Dataset(data_vars={"temperature": da_temperature, "pressure": da_pressure})
ds



import sys
len(sys.argv)

sys.argv

sys.exit(1)