# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Image decoding for explicitly marked remote-sensing ChatML records."""

import logging
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image

AIROUND_RGB_POLICY = "airound_gdal_B4_B3_B2_reflectance_clip01_uint8_v1"
NASC_RGB_POLICY = "nasc_tg2_WIS_V8_V10_V12_per_channel_p2_p98_uint8_v1"


def open_nasc_tg2(path):
    """Render WIS bands V8/V10/V12 (665/565/490 nm) as RGB.

    Band definition: https://www.scitepress.org/Papers/2018/74320/74320.pdf
    These TIFFs have no band descriptions; the adapter assumes native WIS order.
    A deterministic 2/98 percentile stretch is our rendering policy, not a
    claim of pixel parity with the separately published RGB dataset.
    """
    import tifffile

    with tifffile.TiffFile(path) as tif:
        data = tif.asarray(maxworkers=1)
        axes = tif.series[0].axes
    if data.ndim != 3 or "S" not in axes:
        raise ValueError(f"Unexpected NaSC_TG2 shape/axes: {data.shape}/{axes}")
    data = np.moveaxis(data, axes.index("S"), -1)
    if data.shape[-1] != 14 or data.dtype != np.uint16:
        raise ValueError(f"Expected 14-band uint16 NaSC_TG2 TIFF: {path}")
    rgb = data[..., [7, 9, 11]].astype(np.float32)
    low, high = np.percentile(rgb, [2, 98], axis=(0, 1))
    # Flat channels remain finite; do not amplify an arbitrary epsilon range.
    span = np.where(high > low, high - low, 1.0)
    rgb = np.clip((rgb - low) / span, 0, 1)
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))


def open_remote_sensing_image(path, source_dataset=None):
    """Keep ordinary PIL behavior; AiRound Sentinel-2 has 13 float32 bands.

    Its GDAL descriptions identify B4/B3/B2, with reflectance stored in [0, 1].
    Select by description rather than guessing channel order. No percentile
    stretch, gamma correction, model change or on-disk image rewrite is applied.
    """
    path = Path(path)
    if source_dataset == "NaSC_TG2" and path.suffix.lower() in (".tif", ".tiff"):
        return open_nasc_tg2(path)
    if source_dataset != "AiRound" or "sentinel-2" not in path.parts:
        image = Image.open(path)
        if source_dataset is None:
            return image
        try:
            image.load()
            return image
        except OSError as exc:
            image.close()
            if "truncated" not in str(exc).lower():
                raise
            data = path.read_bytes()
            if not data.startswith(b"\xff\xd8") or data.endswith(b"\xff\xd9"):
                raise
            # Some source JPEGs (including files named .png) lack only EOI.
            # Retry a strict decode with the marker appended in memory.
            repaired = Image.open(BytesIO(data + b"\xff\xd9"))
            repaired.load()
            logging.getLogger(__name__).warning("JPEG_EOI_REPAIR %s", path)
            return repaired

    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        metadata = page.tags.get(42112)
        if metadata is None:
            raise ValueError(f"AiRound image has no GDAL band descriptions: {path}")
        xml = ElementTree.fromstring(metadata.value.rstrip("\x00"))
        bands = {
            item.text: int(item.attrib["sample"])
            for item in xml.findall("Item")
            if item.attrib.get("name") == "DESCRIPTION" and "sample" in item.attrib
        }
        if not all(band in bands for band in ("B4", "B3", "B2")):
            raise ValueError(f"AiRound image lacks RGB band descriptions: {path}")
        data = tif.asarray(maxworkers=1)
        axes = tif.series[0].axes
        if data.ndim != 3 or "S" not in axes:
            raise ValueError(f"Unexpected AiRound TIFF shape/axes: {data.shape}/{axes}")
        data = np.moveaxis(data, axes.index("S"), -1)
        rgb = data[..., [bands["B4"], bands["B3"], bands["B2"]]]
        if not np.isfinite(rgb).all():
            raise ValueError(f"Non-finite AiRound RGB reflectance: {path}")
        if not np.issubdtype(rgb.dtype, np.floating):
            raise ValueError(f"Expected floating AiRound reflectance, got {rgb.dtype}: {path}")
        return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
