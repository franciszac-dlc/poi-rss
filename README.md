# Overview

This is a repository that contains POI (Point of interest) recommenders and diversifiers.


# Setup


To set up first start with running the Makefile.

	make

Download Yelp Open Dataset (https://www.yelp.com/dataset) and put it files in data folder. Also download https://www.yelp.com/developers/documentation/v3/all_category_list/categories.json and put it in data directory.

After that, it's possible to generate the pre-processed datasets that we use to do evaluations with the following command:

	cd algorithms/application
	python datasetgen.py

# Citation

Please if this code is useful in your research consider citing the following papers:

	@article{werneck2021systematic,
	  title={A systematic mapping on POI recommendation: Directions, contributions and limitations of recent studies},
	  author={Werneck, Heitor and Silva, N{\'\i}collas and Carvalho, Matheus and Pereira, Adriano CM and Mour{\~a}o, Fernando and Rocha, Leonardo},
	  journal={Information Systems},
	  pages={101789},
	  year={2021},
	  publisher={Elsevier}
	}
	@article{werneck2021effective,
	  title={Effective and diverse POI recommendations through complementary diversification models},
	  author={Werneck, Heitor and Santos, Rodrigo and Silva, N{\'\i}collas and Pereira, Adriano CM and Mour{\~a}o, Fernando and Rocha, Leonardo},
	  journal={Expert Systems with Applications},
	  volume={175},
	  pages={114775},
	  year={2021},
	  publisher={Elsevier}
	}

# Data file reference

## Downloaded files

## Generated files (per city)

### Filters

```
- Drop user-poi duplicates
- Drop POIs with < 5 users visited
- Drop users with < 20 visits in the city
```

### `user/id/<city>.pkl`

```
{
	user_id: user_ndx
}
```

### `poi/id/<city>.pkl`

```
{
	poi_id: poi_ndx
}
```

### `poi_full/<city>.pkl`

```
{
	poi_ndx: {
		{
			'categories': category_normalization(
				poi_data[poi_id]['categories']
			)
		}
	}
}
```

### `poi/<city>.pkl`

```
{
	poi_ndx: {
		{
			'categories': category_filter(
				poi_data[poi_id]['categories']
			)
		}
	}
}
```

### `neighbor/<city>.pkl`

```
{
	poi_ndx: list(
		poi_coos_balltree.query_radius(
			[pois_coos[lid]],
			geocat_constants.NEIGHBOR_DISTANCE/earth_radius
		)[0]
	)
}
```

### `user/friend/<city>.pkl`

```
{
	user_ndx: [user_ndx friends]
}
```

### `user/<city>.pkl`

```
{
	user_ndx: { user data }
}
```

### `checkin/<city>.pkl`

```
[
	{
		'user_id': user_ndx,
		'poi_id': poi_ndx,
		'date': pd.Datetime,
	}
]
```

### `checkin/<train,test>/<city>.pkl`

Note that this step also does per-user splits and filters to avoid data leakage.

[
	{
		'user_id': user_ndx,
		'poi_id': poi_ndx,
		'date': pd.Datetime,
	}
]
