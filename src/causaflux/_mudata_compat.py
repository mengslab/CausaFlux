"""Minimal MuData/AnnData compatibility layer for constrained build environments.

CausaFlux uses the official :mod:`anndata` and :mod:`mudata` packages whenever
available.  This module implements only the small object surface required by the
bundled synthetic demonstration.  Its writer follows the public AnnData v0.2.0
element encodings and MuData v0.1.0 HDF5 hierarchy, so artifacts created in a
minimal build environment retain the standard ``.h5mu`` structure.

It is intentionally not a replacement for the scverse APIs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_ARRAY_VERSION = "0.2.0"
_MAPPING_VERSION = "0.1.0"
_MUDATA_VERSION = "0.1.0"
_ANNDATA_VERSION = "0.1.0"


def _set_encoding(obj: h5py.Group | h5py.Dataset, kind: str, version: str) -> None:
    obj.attrs["encoding-type"] = kind
    obj.attrs["encoding-version"] = version


def _string_array(values: Any) -> np.ndarray:
    return np.asarray([str(v) for v in np.asarray(values).reshape(-1)], dtype=object)


def _write_array(parent: h5py.Group, key: str, values: Any) -> h5py.Dataset:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U", "S"}:
        dataset = parent.create_dataset(key, data=_string_array(array), dtype=_STRING_DTYPE)
        _set_encoding(dataset, "string-array", _ARRAY_VERSION)
    else:
        dataset = parent.create_dataset(key, data=array)
        _set_encoding(dataset, "array", _ARRAY_VERSION)
    return dataset


def _write_scalar(parent: h5py.Group, key: str, value: Any) -> h5py.Dataset:
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        dataset = parent.create_dataset(key, data=str(value), dtype=_STRING_DTYPE)
        _set_encoding(dataset, "string", _ARRAY_VERSION)
    else:
        dataset = parent.create_dataset(key, data=value)
        _set_encoding(dataset, "numeric-scalar", _ARRAY_VERSION)
    return dataset


def _write_mapping(parent: h5py.Group, key: str, mapping: Mapping[str, Any]) -> h5py.Group:
    group = parent.create_group(key)
    _set_encoding(group, "dict", _MAPPING_VERSION)
    for name, value in mapping.items():
        _write_element(group, str(name), value)
    return group


def _write_element(parent: h5py.Group, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        _write_mapping(parent, key, value)
    elif isinstance(value, pd.DataFrame):
        _write_frame(parent, key, value)
    elif sparse.issparse(value):
        _write_sparse(parent, key, value)
    elif np.isscalar(value):
        _write_scalar(parent, key, value)
    else:
        _write_array(parent, key, value)


def _write_frame(parent: h5py.Group, key: str, frame: pd.DataFrame) -> h5py.Group:
    group = parent.create_group(key)
    _set_encoding(group, "dataframe", "0.2.0")
    index_key = "_index"
    group.attrs["_index"] = index_key
    group.attrs.create("column-order", np.asarray([str(c) for c in frame.columns], dtype=object), dtype=_STRING_DTYPE)
    _write_array(group, index_key, frame.index.astype(str).to_numpy())
    for column in frame.columns:
        series = frame[column]
        # Avoid nullable/categorical encodings in this deliberately small writer.
        if pd.api.types.is_bool_dtype(series.dtype):
            values = series.fillna(False).to_numpy(dtype=bool)
        elif pd.api.types.is_numeric_dtype(series.dtype):
            values = series.to_numpy()
        else:
            values = series.fillna("").astype(str).to_numpy()
        _write_array(group, str(column), values)
    return group


def _write_sparse(parent: h5py.Group, key: str, matrix: sparse.spmatrix) -> h5py.Group:
    matrix = matrix.tocsr()
    group = parent.create_group(key)
    _set_encoding(group, "csr_matrix", "0.1.0")
    group.attrs["shape"] = np.asarray(matrix.shape, dtype=np.int64)
    _write_array(group, "data", matrix.data)
    _write_array(group, "indices", matrix.indices)
    _write_array(group, "indptr", matrix.indptr)
    return group


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str)
    return value


def _read_array(dataset: h5py.Dataset) -> np.ndarray:
    values = dataset[()]
    if getattr(values, "dtype", None) is not None and values.dtype.kind in {"O", "S", "U"}:
        flat = [_decode(v) for v in np.asarray(values).reshape(-1)]
        return np.asarray(flat, dtype=object).reshape(np.asarray(values).shape)
    return values


def _read_frame(group: h5py.Group) -> pd.DataFrame:
    index_key = _decode(group.attrs.get("_index", "_index"))
    index = pd.Index(_read_array(group[index_key]).astype(str), name=None)
    order = group.attrs.get("column-order", [k for k in group.keys() if k != index_key])
    columns = [_decode(v) for v in list(order)]
    data: dict[str, Any] = {}
    for column in columns:
        data[str(column)] = _read_element(group[str(column)])
    return pd.DataFrame(data, index=index)


def _read_sparse(group: h5py.Group) -> sparse.csr_matrix:
    shape = tuple(int(v) for v in group.attrs["shape"])
    return sparse.csr_matrix(
        (_read_array(group["data"]), _read_array(group["indices"]), _read_array(group["indptr"])),
        shape=shape,
    )


def _read_element(item: h5py.Group | h5py.Dataset) -> Any:
    encoding = _decode(item.attrs.get("encoding-type", ""))
    if isinstance(item, h5py.Dataset):
        value = item[()]
        if encoding == "string":
            return _decode(value)
        if encoding in {"string-array", "array", ""}:
            return _read_array(item)
        return _decode(value)
    if encoding == "dataframe":
        return _read_frame(item)
    if encoding in {"csr_matrix", "csc_matrix"}:
        return _read_sparse(item)
    if encoding in {"dict", ""}:
        return {name: _read_element(child) for name, child in item.items()}
    return {name: _read_element(child) for name, child in item.items()}


class AnnData:
    """Narrow in-memory representation used only when :mod:`anndata` is absent."""

    def __init__(self, X, obs: pd.DataFrame | None = None, var: pd.DataFrame | None = None):
        self.X = X
        n_obs, n_vars = X.shape
        self.obs = obs.copy() if obs is not None else pd.DataFrame(index=[str(i) for i in range(n_obs)])
        self.var = var.copy() if var is not None else pd.DataFrame(index=[str(i) for i in range(n_vars)])
        self.obsm: dict[str, Any] = {}
        self.uns: dict[str, Any] = {}

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def var_names(self):
        return self.var.index

    @property
    def n_obs(self):
        return int(self.X.shape[0])

    @property
    def n_vars(self):
        return int(self.X.shape[1])


class MuData:
    """Narrow multimodal container used only when :mod:`mudata` is absent."""

    def __init__(self, modalities: Mapping[str, AnnData]):
        self.mod = dict(modalities)
        first = next(iter(self.mod.values()))
        self.obs = pd.DataFrame(index=first.obs_names.copy())
        self.obsm: dict[str, Any] = {}
        self.uns: dict[str, Any] = {}

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def n_obs(self):
        return len(self.obs)

    @property
    def var(self) -> pd.DataFrame:
        pieces = []
        for modality, adata in self.mod.items():
            part = adata.var.copy()
            if "modality" not in part.columns:
                part["modality"] = modality
            pieces.append(part)
        return pd.concat(pieces, axis=0) if pieces else pd.DataFrame()

    @property
    def n_vars(self) -> int:
        return int(sum(adata.n_vars for adata in self.mod.values()))

    def write_h5mu(self, path: str | Path) -> None:
        """Write a MuData v0.1.0 / AnnData-element-compatible HDF5 file."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w", userblock_size=512) as handle:
            handle.attrs["axis"] = 0
            handle.attrs["encoding-type"] = "MuData"
            handle.attrs["encoding-version"] = _MUDATA_VERSION
            handle.attrs["encoder"] = "causaflux"
            handle.attrs["encoder-version"] = "1.7.0"

            _write_frame(handle, "obs", self.obs)
            top_var = self.var
            _write_frame(handle, "var", top_var)
            _write_mapping(handle, "obsm", self.obsm)
            for key in ("varm", "obsp", "varp"):
                _write_mapping(handle, key, {})
            _write_mapping(handle, "uns", self.uns)

            obsmap: dict[str, np.ndarray] = {}
            varmap: dict[str, np.ndarray] = {}
            offset = 0
            for modality, adata in self.mod.items():
                # The CausaFlux data contract keeps a common row index across modalities.
                positions = pd.Series(np.arange(1, adata.n_obs + 1), index=adata.obs_names.astype(str))
                obsmap[modality] = positions.reindex(self.obs.index.astype(str), fill_value=0).to_numpy(dtype=np.uint32)
                mapping = np.zeros(len(top_var), dtype=np.uint32)
                mapping[offset : offset + adata.n_vars] = np.arange(1, adata.n_vars + 1, dtype=np.uint32)
                varmap[modality] = mapping
                offset += adata.n_vars
            _write_mapping(handle, "obsmap", obsmap)
            _write_mapping(handle, "varmap", varmap)

            mod_group = handle.create_group("mod")
            mod_group.attrs.create("mod-order", np.asarray(list(self.mod), dtype=object), dtype=_STRING_DTYPE)
            for name, adata in self.mod.items():
                group = mod_group.create_group(name)
                group.attrs["encoding-type"] = "anndata"
                group.attrs["encoding-version"] = _ANNDATA_VERSION
                group.attrs["encoder"] = "causaflux"
                group.attrs["encoder-version"] = "1.7.0"
                _write_element(group, "X", adata.X)
                _write_frame(group, "obs", adata.obs)
                _write_frame(group, "var", adata.var)
                _write_mapping(group, "obsm", getattr(adata, "obsm", {}))
                for key in ("varm", "obsp", "varp", "layers"):
                    _write_mapping(group, key, {})
                _write_mapping(group, "uns", getattr(adata, "uns", {}))

        # MuData readers use the 512-byte HDF5 user block as a format marker.
        with path.open("r+b") as stream:
            marker = b"MuData (format-version=0.1.0;creator=causaflux;creator-version=1.7.0)"
            stream.write(marker)
            stream.write(b"\0" * (512 - len(marker)))


def _read_legacy_h5mu(handle: h5py.File) -> MuData:
    """Read the private pre-release compatibility layout for backward migration."""

    def decode_legacy_frame(group: h5py.Group) -> pd.DataFrame:
        index = pd.Index([_decode(x) for x in group["_index"][()]], name=group.attrs.get("index_name", "_index"))
        data = {}
        for name, item in group.items():
            if name == "_index":
                continue
            values = item["values"][()]
            if item.attrs.get("kind") == "string":
                values = [_decode(x) for x in values]
            data[name] = values
        return pd.DataFrame(data, index=index)

    modalities = {}
    for name, group in handle["mod"].items():
        x = group["X"]
        if x.attrs["kind"] == "csr":
            shape = tuple(int(v) for v in x.attrs["shape"])
            matrix = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=shape)
        else:
            matrix = x["values"][()]
        modalities[name] = AnnData(matrix, decode_legacy_frame(group["obs"]), decode_legacy_frame(group["var"]))
    mdata = MuData(modalities)
    mdata.obs = decode_legacy_frame(handle["obs"])
    import json

    raw = handle["uns_json"][()]
    mdata.uns = json.loads(_decode(raw))
    return mdata


def read_h5mu(path: str | Path) -> MuData:
    """Read standard CausaFlux demonstration H5MU or migrate the pre-release layout."""

    with h5py.File(path, "r") as handle:
        if _decode(handle.attrs.get("encoding-type", "")) == "causaflux-mudata-compat":
            return _read_legacy_h5mu(handle)
        modalities: dict[str, AnnData] = {}
        mod_group = handle["mod"]
        order = mod_group.attrs.get("mod-order", list(mod_group.keys()))
        for raw_name in list(order):
            name = str(_decode(raw_name))
            group = mod_group[name]
            matrix = _read_element(group["X"])
            obs = _read_frame(group["obs"])
            var = _read_frame(group["var"])
            modalities[name] = AnnData(matrix, obs, var)
        mdata = MuData(modalities)
        mdata.obs = _read_frame(handle["obs"])
        if "obsm" in handle:
            mdata.obsm = _read_element(handle["obsm"])
        if "uns" in handle:
            mdata.uns = _read_element(handle["uns"])
        return mdata
