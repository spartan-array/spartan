from tile_py import Tile, TilePy, from_data, from_shape

TYPE_DENSE = Tile.TILE_DENSE
TYPE_MASKED = Tile.TILE_MASKED
TYPE_SPARSE = Tile.TILE_SPARSE

if Tile != TilePy:
  Tile = TilePy
