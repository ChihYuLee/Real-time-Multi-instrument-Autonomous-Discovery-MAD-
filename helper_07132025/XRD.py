# Standard library imports
import io
import pickle
import re
import warnings
from collections import namedtuple
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third party imports
import fabio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skimage.measure
import skimage.restoration as restoration
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon, Wedge
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.special import erfc
from scipy.stats import binned_statistic_2d
from skimage.draw import circle_perimeter, disk


def save_pickle(obj, file):
    with Path(file).open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(file):
    with Path(file).open("rb") as f:
        return pickle.load(f)


def _create_figure(ax=None, **subplots_kargs):
    if ax is None:
        return plt.subplots(**subplots_kargs)
    else:
        return None, ax


def ternary2cartesian(a, b, c):
    # translate the data to cartesian corrds
    x = 0.5 * (2.0 * b + c) / 1  # 1 could be 100 if comps is in percentage
    y = 0.5 * np.sqrt(3) * c / 1
    return x, y


def tt2q(x, wavelength=None):
    x = np.deg2rad(x)
    x = 4 * np.pi * np.sin(x / 2)
    if wavelength is not None:
        x = x / wavelength
    return x


def q2tt(x, wavelength=None):
    if wavelength is not None:
        x = x * wavelength

    x = np.arcsin(x / 4 / np.pi)
    x = np.rad2deg(x) * 2
    return x


def q2d(x):
    return (2 * np.pi) / x


def d2q(x):
    return (2 * np.pi) / x


# misc
def cartician2polar(xy, center):
    _xy = xy - center
    rho = np.linalg.norm(_xy, axis=1)
    phi = np.arctan2(_xy[:, 1], _xy[:, 0])
    phi = ((phi / np.pi + 2) % 2) * np.pi
    return np.stack((rho, phi), axis=1)


def polar2cartician(rp, center):
    center = np.array(center)
    if len(center.shape) == 1:
        center = center[None, :]
    rho = rp[:, 0]
    phi = rp[:, 1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.stack((x, y), axis=1) + center


def convert_ws(ws: List, ws_label: str, axis_type: str, wavelength: float):
    """
    a convertion between 2θ and q. Here we assume wavelengths is in Å

    Arguments:
        axis_type (str) : could be "tt" (2theta) or "q" (Q space)
    """
    if axis_type == "q" and ws_label == "2θ":
        # ws = np.sin(np.deg2rad(ws/2)) * 4 * np.pi / wavelength
        ws = tt2q(ws, wavelength=wavelength)
        ws_label = "Q"
        ws_unit = "1/Å"

    elif axis_type == "tt" and ws_label == "Q":
        # ws = np.rad2deg( np.arcsin(ws / (4 * np.pi / wavelength) ) ) * 2
        ws = q2tt(ws, wavelength=wavelength)
        ws_label = "2θ"
        ws_unit = "°"
    else:
        raise ValueError("Argument axis_type must be one of ['tt', 'q']")

    return ws, ws_label, ws_unit


class X00File:
    META_TYPE = {
        "FileName": str,
        "FileDateTime": str,
        "Sample": str,
        "Reflection": str,
        "Wavelength": float,
        "GenkVmA": str,
        "Omega": float,
        "TwoTheta": float,
        "X": float,
        "Y": float,
        "Phi": float,
        "Psi": float,
        "ScanType": str,
        "ScanAxis": str,
        "FirstAngle": float,
        "ScanRange": float,
        "StepWidth": float,
        "TimePerStep": float,
        "NrOfData": int,
    }

    def __init__(self, path: Union[io.StringIO, io.FileIO, str]) -> None:
        if isinstance(path, io.StringIO) or isinstance(path, io.FileIO):
            f = path
        else:
            f = open(path, "r")
        with f:
            self._parse_file(f)

    def _parse_file(self, f):
        self.file_type = f.readline().strip()
        # print(self.file_type)
        metatexts = []

        while True:
            line = f.readline().strip()
            if line.startswith("ScanData"):
                break
            else:
                metatexts.append(line)

        # print(metatexts)
        self._parse_metas(metatexts)
        self.y = np.loadtxt(io.StringIO(f.read()))

    def _parse_metas(self, metatexts):
        self.metas = {}
        for meta in metatexts:
            name, value = meta.split(",", 1)
            name = name.strip()
            value = value.strip()
            if name in self.META_TYPE:
                value = self.META_TYPE[name](value)
            self.metas[name] = value

        try:
            self.x = np.arange(
                self.metas["FirstAngle"],
                self.metas["FirstAngle"]
                + self.metas["ScanRange"]
                + self.metas["StepWidth"] / 2,
                self.metas["StepWidth"],
            )
        except Exception as e:
            warnings.warn(f"Cannot parse the twotheta information.\n{e}")


class XYFile:
    def __init__(self, path: Union[io.StringIO, io.FileIO, str]) -> None:
        if isinstance(path, io.StringIO) or isinstance(path, io.FileIO):
            f = path
        else:
            f = open(path, "r")
        with f:
            x, y, comments = self._parse_file(f)

        self._parse_comments(comments)

    def _parse_file(self, file):
        self.comments = file.readline()
        xy = np.loadtxt(file)
        self.x = xy[:, 0]
        self.y = xy[:, 1]

        return self.x, self.y, self.comments

    def _parse_comments(self, comments):
        comments = comments[1:]
        _comments = re.findall(r'([\w]+): "(.*?)"', comments)
        self.headers = {}
        for item in _comments:
            self.headers[item[0]] = item[1]

        return self.headers


class PltFile:
    _comment_pat: str = r"!@!!?"

    def _is_comment(self, line: str):
        return re.match(self._comment_pat, line) is not None

    def _find_by_keyword(self, comments: str, kw: str, splitor: str = ":"):
        for c in comments:
            if c.startswith(kw):
                return c.split(splitor)[1].strip()

    def _parse_file_type(self, comments: str):
        return self._find_by_keyword(comments, "GADDS PLOTSO FILE")

    def _parse_name(self, comments: str):
        return self._find_by_keyword(comments, "Title/SampleName")

    def _parse_user(self, comments: str):
        return self._find_by_keyword(comments, "User")

    def _parse_site(self, comments: str):
        return self._find_by_keyword(comments, "Site")

    def _parse_xy_label(self, comments: str):
        return comments[-2], comments[-1]

    def _parse_experiment(self, comments: str):
        for c in comments:
            if c.startswith("Wavelengths"):
                wavelengths = re.split(r"\s+", c)[1:]
                wavelengths = tuple(map(float, wavelengths))
            if c.startswith("Frame angles"):
                c = c.split(":", maxsplit=1)[1]
                pairs = re.findall(r"(\w+)\s*:\s*([\w\.]+)", c)
                frame_angles = {p[0]: float(p[1]) for p in pairs}
            if c.startswith("Integration range"):
                c = c.split(":", maxsplit=1)[1]
                ranges = re.findall(r"(\w+)\s*:\s*([\-\d\.]+)\s*to\s*([\-\d\.]+)", c)
                integration_range = {p[0]: (float(p[1]), float(p[2])) for p in ranges}
            if c.startswith("Integration method"):
                integration_method = c.split(":")[1].strip()

        return wavelengths, frame_angles, integration_range, integration_method

    def _parse_comments(self, comments: str):
        self.file_type = self._parse_file_type(comments)
        self.name = self._parse_name(comments)
        self.user = self._parse_user(comments)
        self.site = self._parse_site(comments)
        wavelengths, frame_angles, integration_range, integration_method = (
            self._parse_experiment(comments)
        )
        self.wavelengths = wavelengths
        self.frame_angles = frame_angles
        self.integration_range = integration_range
        self.integration_method = integration_method
        self.xlabel, self.ylabel = self._parse_xy_label(comments)

    def _parse_file(self, f: Union[io.StringIO, io.FileIO]):

        x = []
        y = []
        comments = []

        for line in f.readlines():
            if self._is_comment(line):
                content = re.split(self._comment_pat, line)[1]
                comments.append(content.strip())

            else:
                xdegree, ycounts = line.split(" ")
                x.append(float(xdegree.strip()))
                y.append(float(ycounts.strip()))

        self.x = np.array(x)
        self.y = np.array(y)
        self.comments = comments
        return self.x, self.y, self.comments

    def __init__(self, path: Union[io.StringIO, io.FileIO, str]):
        if isinstance(path, io.StringIO) or isinstance(path, io.FileIO):
            f = path
        else:
            f = open(path, "r")
        with f:
            x, y, comments = self._parse_file(f)
        self._parse_comments(comments)

    def __repr__(self):
        return super().__repr__() + f"\n #x: {len(self.x)} #y: {len(self.y)}"


class WDSSummaryFile:
    def __init__(self, path: Union[io.StringIO, io.FileIO, str]):
        if isinstance(path, io.StringIO) or isinstance(path, io.FileIO):
            f = path
        else:
            f = open(path, "r")
        with f:
            self._parse_file(f)

    def _parse_header(self, file):
        line = file.readline()
        self.header = []
        assert line.strip() == ""

        for line in file:
            if line.strip() == "":
                break
            items = [e.strip() for e in line.split(",")]
            self.header += items
        return self.header

    def _parse_body(self, file):
        self.body = pd.read_csv(file, index_col=0)
        columns = self.body.columns
        self.body = self.body.rename(columns={c: c.strip() for c in columns})
        self.elements = list(self.body.columns)[0:-2]

    def _parse_file(self, file):
        self._parse_header(file)
        self._parse_body(file)

    def __repr__(self):
        return (
            super().__repr__()
            + f"\n #samples: {len(self.body)} #elements: {self.elements}"
        )


@dataclass
class DetectorSpec:
    name: str
    type: str
    shape: str
    sample_detector_dist: float
    pixel_size: float
    pixel_per_cm: float
    frame_size: float
    frame_size_pixel: int
    pitch: float
    roll: float
    yaw: float
    beam_center: list
    effective_radius: float


_detector_spec = DetectorSpec(
    name="VANTEC-500",
    type="MIKROGAP",
    shape="Circular",
    pixel_size=0.0068,
    pixel_per_cm=147.06,
    frame_size=13.926,
    frame_size_pixel=2048,
    pitch=0.00,
    roll=0.00,
    yaw=0.00,
    sample_detector_dist=32.4,
    beam_center=[1002.75, 1012.00],
    effective_radius=800,
)


def pidx2coor(x_i, y_i, x_size, y_size, center=None):
    if center is None:
        center = [0, 0]
    # minus sign to reverse the axis
    x_loc = -(x_i - center[0]) * x_size
    y_loc = (y_i - center[1]) * y_size
    return x_loc, y_loc


def change_origin(x, y, new_ori):
    return x - new_ori[0], y - new_ori[1]


def plane2real(x_loc, y_loc, basis_transformation, translation=None):
    basis_transformation = np.expand_dims(
        basis_transformation, list(range(len(x_loc.shape)))
    )
    plane_coors = np.stack((x_loc, y_loc), axis=-1)
    _coors = np.expand_dims(plane_coors, axis=-1)
    coors_real = (basis_transformation @ _coors)[..., 0]
    if translation is not None:
        translation = np.expand_dims(translation, list(range(len(x_loc.shape))))
        coors_real += translation
    return coors_real


def detectaor_not_implemented_warning(detector_spec: DetectorSpec):
    if abs(detector_spec.pitch) > 1e-5 or abs(detector_spec.roll) > 1e-5 or abs(detector_spec.yaw) > 1e-5:
        warnings.warn("Detector spec with non-zero pitch, roll, or yaw is not implemented yet.")

class Frame:
    """
    Now only Work For Bruker Image.
    """

    def __init__(self, frame: Union[fabio.brukerimage.BrukerImage, Path, str], detector_spec: DetectorSpec, omit_theta: bool = True):

        if isinstance(frame, Path) or isinstance(frame, str):
            frame = fabio.open(frame)

        self.frame = frame  # fabio frame
        self.omit_theta = omit_theta
        self.detector_spec = detector_spec
        detectaor_not_implemented_warning(detector_spec)

        self._parse_frame()
        self._get_incident_beam()
        # incident beam direction
        # useful when used to calculate the 2theta and chi

    def get_effective_mask(self):
        mask = np.zeros_like(self.frame.data, dtype=bool)
        # our beam_center is in xy coordinates not in row and column coordinates

        rr, cc = disk(
            self.detector_spec.beam_center[::-1],
            radius=self.detector_spec.effective_radius,
            shape=self.frame.data.shape,
        )
        mask[rr, cc] = 1
        return mask

    def get_effective_mask_edge(self):
        rr, cc = circle_perimeter(
            r=round(self.detector_spec.beam_center[1]),
            c=round(self.detector_spec.beam_center[0]),
            radius=round(self.detector_spec.effective_radius),
            shape=self.frame.data.shape,
        )

        cc_ = cc - round(self.detector_spec.beam_center[0])
        rr_ = rr - round(self.detector_spec.beam_center[1])
        r = round(self.detector_spec.effective_radius)
        theta = np.arccos(np.clip(cc_ / r, -1, 1))
        theta[np.sign(rr_) > 0] = np.pi * 2 - theta[np.sign(rr_) > 0]
        idx = np.argsort(theta)

        return rr[idx], cc[idx]

    def _get_incident_beam(self):
        self.incident_beam_dir = np.array(
            [
                0,
                np.cos(np.deg2rad(-(self.beam_center_ttheta - self.beam_center_omega))),
                np.sin(np.deg2rad(-(self.beam_center_ttheta - self.beam_center_omega))),
            ]
        )
        return self.incident_beam_dir

    def _parse_frame(self):
        goniometer_angles = list(
            map(float, self.frame.header["ANGLES"].split())
        )  # "    "

        self.beam_center_ttheta = goniometer_angles[0]
        self.beam_center_omega = goniometer_angles[1]
        self.beam_center_theta = self.beam_center_ttheta - self.beam_center_omega

        if self.omit_theta:
            self.beam_center_omega = self.beam_center_ttheta
            self.beam_center_theta = 0
            self.raw_angles = goniometer_angles

        # Actual goniometer linear axes @ end of frame. (X, Y, Z, Aux)
        self.stage_pos = list(
            map(float, self.frame.header["ENDING2"].split())
        )  # "    "

    def get_3d_coor(self):
        # how could we calculate the 2theta along the horizontal direction?
        # 2theta grow gradually from right to left as you can see the radius of the xray cone increase

        beam_center_omega = self.beam_center_omega

        a1 = np.array(
            [
                0,
                -np.sin(np.deg2rad(beam_center_omega)),
                np.cos(np.deg2rad(beam_center_omega)),
            ]
        )
        a2 = np.array([1, 0, 0])
        basis_trans = np.stack((a1, a2), axis=1)
        translation = np.array(
            [
                0,
                np.cos(np.deg2rad(beam_center_omega))
                * self.detector_spec.sample_detector_dist,
                np.sin(np.deg2rad(beam_center_omega))
                * self.detector_spec.sample_detector_dist,
            ]
        )

        # pixel index
        # frame.data.shape

        # x
        x_i = np.arange(self.frame.data.shape[1])
        y_i = np.arange(self.frame.data.shape[0])

        self._x_i, self._y_i = np.meshgrid(x_i, y_i)

        # compute the x, y coor with beam center as origin
        x_d, y_d = pidx2coor(
            self._x_i,
            self._y_i,
            self.detector_spec.pixel_size,
            self.detector_spec.pixel_size,
            self.detector_spec.beam_center,
        )

        self.pixel_coors = plane2real(x_d, y_d, basis_trans, translation)

        return self.pixel_coors

    def get_pixel_angles(self):
        px = self.pixel_coors[:, :, 0]
        py = self.pixel_coors[:, :, 1]
        pz = self.pixel_coors[:, :, 2]
        # chis = np.rad2deg(np.arctan(px / (pz+1e-9) ))

        # omegas = np.rad2deg(np.arctan( np.sqrt((pz**2 + px**2)/ (py+1e-9)**2) ) )
        # omegas = np.rad2deg(np.arctan( np.sqrt((pz**2 + px**2))/ py ) )
        # tths = omegas + beam_center_ttheta - beam_center_omega
        pixel_coors_norm = np.linalg.norm(self.pixel_coors, axis=-1)
        if self.omit_theta:
            # could simplify computation since we assume incident beam is pointing at y direction
            cos_tths = self.pixel_coors[:, :, 1] / pixel_coors_norm
        else:
            cos_tths = np.squeeze(
                self.incident_beam_dir[None, None, None, :]
                @ self.pixel_coors[:, :, :, None],
                axis=(-2, -1),
            ) / (pixel_coors_norm)
            # cos_tths = np.squeeze(self.incident_beam_dir[None,None,None,:] @ self.pixel_coors[:,:,:,None], axis=(-2, -1)) / ( np.linalg.norm(self.incident_beam_dir) * pixel_coors_norm )

        self.tths = tths = np.rad2deg(np.arccos(cos_tths))

        # should be prependicular to k all the time
        if self.omit_theta:
            # beam center arm is always in z axis
            # omega always equal to tth
            # beam_center_arm = [0,0,1]
            # chi_arm = [pixel_coors[0], 0, pixel_coors[2]]
            chi_arm = self.pixel_coors[:, :, [0, 2]]
            cos_chis = chi_arm[:, :, -1] / (np.linalg.norm(chi_arm, axis=-1))
        else:
            coor_projs = (cos_tths * pixel_coors_norm)[
                :, :, None
            ] * self.incident_beam_dir[None, None, :]
            chi_arm = self.pixel_coors - coor_projs
            omegas = tths - (self.beam_center_ttheta - self.beam_center_omega)
            beam_center_arm_y = np.cos(np.deg2rad(90 - tths + omegas))
            beam_center_arm_z = np.sin(np.deg2rad(90 - tths + omegas))
            beam_center_arm = np.stack(
                (
                    np.zeros_like(beam_center_arm_y),
                    beam_center_arm_y,
                    beam_center_arm_z,
                ),
                axis=-1,
            )

            cos_chis = np.squeeze(
                beam_center_arm[:, :, None, :] @ chi_arm[:, :, :, None], axis=(-2, -1)
            ) / (np.linalg.norm(chi_arm, axis=-1))
            # cos_chis = np.squeeze(beam_center_arm[:,:,None,:] @ chi_arm[:,:,:,None], axis=(-2, -1)) / ( np.linalg.norm(beam_center_arm, axis=-1) * np.linalg.norm(chi_arm, axis=-1) )
        self.chis = chis = np.sign(chi_arm[:, :, 0]) * np.rad2deg(
            np.arccos(np.clip(cos_chis, -1, 1))
        )

        return self.tths, self.chis

    def get_pixel_angleareas(self):

        # # delta theta per pixel
        # _delta_tths = self.tths[:, :-1] - self.tths[:, 1:]
        # delta_tths = np.zeros_like(self.tths)
        # delta_tths[:, 1:] = _delta_tths
        # delta_tths[:, 0] = delta_tths[:, 1]

        # # delta chis per pixel
        # _delta_chis = self.chis[1:, :] - self.chis[:-1, :]
        # delta_chis = np.zeros_like(self.chis)
        # delta_chis[1:, :] = _delta_chis
        # delta_chis[0, :] = delta_tths[1, :]

        # self.pixel_angleareas = delta_tths * delta_chis
        # return self.pixel_angleareas

        # use the area of quadrilateral formula to get the pixel angle area
        self.pixel_angleareas = np.zeros_like(self.tths)
        tcs = np.stack((self.tths, self.chis), axis=-1)
        p1 = tcs[:-1, :-1]
        p2 = tcs[1:, :-1]
        p3 = tcs[:-1, 1:]
        p4 = tcs[1:, 1:]
        a = p1 - p3
        a_n = np.linalg.norm(a, axis=-1)
        b = p1 - p2
        b_n = np.linalg.norm(b, axis=-1)
        c = p2 - p4
        c_n = np.linalg.norm(c, axis=-1)
        d = p3 - p4
        d_n = np.linalg.norm(d, axis=-1)
        cth1 = np.squeeze(a[:, :, None, :] @ b[:, :, :, None], axis=(-2, -1)) / (
            np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        )
        cth2 = np.squeeze(c[:, :, None, :] @ d[:, :, :, None], axis=(-2, -1)) / (
            np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        )
        th1 = np.arccos(cth1)
        th2 = np.arccos(cth2)
        s = (a_n + b_n + c_n + d_n) / 2

        area_sq = (s - a_n) * (s - b_n) * (s - c_n) * (
            s - d_n
        ) - a_n * b_n * c_n * d_n * (np.cos((th1 + th2) / 2)) ** 2
        qa = np.sqrt(np.sqrt(area_sq))

        self.pixel_angleareas[:-1, :-1] = qa
        self.pixel_angleareas[-1, :-1] = qa[-1, :]
        self.pixel_angleareas[:-1, -1] = qa[:, -1]
        self.pixel_angleareas[-1, -1] = qa[-1, -1]
        return self.pixel_angleareas

    def plot_pixel_angles(self, vmin=0, vmax=None):
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

        ax = axs[0]
        CS = ax.contour(
            self._x_i,
            self._y_i,
            self.tths,
            levels=15,
            zorder=100,
            colors="white",
            linewidths=1,
        )
        ax.clabel(CS, inline=True, fontsize=10, fmt="%1.1f")

        ax.imshow(
            self.frame.data[:, :],  # first dim is col (y) then is row (x)
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            #  alpha=0.5,
            zorder=1,
        )

        fh, fw = self.frame.data.shape
        ax.set_xlim(-fw * 0.05, fw + fw * 0.05)
        ax.set_ylim(-fh * 0.05, fh + fh * 0.05)

        ax = axs[1]
        CS = ax.contour(
            self._x_i,
            self._y_i,
            self.chis,
            levels=15,
            zorder=100,
            colors="white",
            linewidths=1,
        )
        ax.clabel(CS, inline=True, fontsize=10, fmt="%1.1f")

        ax.imshow(
            self.frame.data[:, :],  # first dim is col (y) then is row (x)
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            #  alpha=0.5,
            zorder=1,
        )

        ax.set_xlim(-fw * 0.05, fw + fw * 0.05)
        ax.set_ylim(-fh * 0.05, fh + fh * 0.05)

        return fig, ax

    def plot_3d_coors(self, downscale=50, cmin=0, cmax=None):

        # https://community.plotly.com/t/trying-to-add-a-png-jpg-image-to-a-3d-surface-graph-r/4192/2
        # directly draw the pattern, not sure how it scale

        fig = go.Figure()

        s = downscale
        x = self.pixel_coors[::s, ::s, 0].ravel()
        y = self.pixel_coors[::s, ::s, 1].ravel()
        z = self.pixel_coors[::s, ::s, 2].ravel()

        _frame = self.frame.data.copy()
        m = _frame.max()
        _frame[:s, :s] = m * 2
        _frame[-s * 2 :, -s * 2 :] = m * 2
        c = skimage.measure.block_reduce(_frame, (s, s), np.mean).ravel()

        if cmax is None:
            cmax = m

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker={"color": c, "size": 4, "cmin": cmin, "cmax": cmax},
                mode="markers",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[self.pixel_coors[0, 0, 0]],
                y=[self.pixel_coors[0, 0, 1]],
                z=[self.pixel_coors[0, 0, 2]],
                mode="markers",
                name="0,0",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[self.pixel_coors[-1, -1, 0]],
                y=[self.pixel_coors[-1, -1, 1]],
                z=[self.pixel_coors[-1, -1, 2]],
                mode="markers",
                name="-1,-1",
            )
        )

        return fig

    def plot_center(self, radius=None):
        fig, ax = plt.subplots()

        ax.imshow(
            self.frame.data,  # first dim is col (y) then is row (x)
            cmap="plasma",
            vmax=1,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        patches = []
        r = 800 if radius is None else radius
        beam_center = self.detector_spec.beam_center
        circle = Circle(
            (beam_center[0] + 10, beam_center[1]), r, ec="white", fill=False
        )
        ax.add_patch(circle)

        # patches.append(circle)
        # p = PatchCollection(patches)
        # ax.add_collection(p)

        ax.scatter(beam_center[0], beam_center[1], s=5, color="white")

    def plot_frame(
        self,
        chi_min=None,
        chi_max=None,
        tths_min=None,
        tths_max=None,
        apply_effective_mask=True,
        vmin=None,
        vmax=None,
    ):
        mask = np.ones(self.frame.data.shape, dtype=bool)

        if chi_min is not None and chi_max is not None:
            chi_mask = (self.chis > chi_min) & (self.chis < chi_max)
            mask = mask & (chi_mask)

        if tths_min is not None and tths_max is not None:
            tths_mask = (self.tths > tths_min) & (self.tths < tths_max)
            mask = mask & (tths_mask)

        if apply_effective_mask:
            c_mask = self.get_effective_mask()
            mask = mask & (c_mask)

        fig, ax = plt.subplots()

        frame_data = self.frame.data.copy()
        frame_data[~mask] = 0
        ax.imshow(frame_data, vmin=vmin, vmax=vmax, cmap="plasma")

        return fig, ax

    def plot_warp_frame(self, vmin=0, vmax=None):
        fig, ax = plt.subplots()

        frame_data = self.warpped_frame.copy()
        ax.imshow(frame_data, vmin=vmin, vmax=vmax, cmap="plasma")

        ax.set_xticks(self.warpped_frame_tths)
        ax.set_xticklabels(self.warpped_frame_tths)

        ax.set_yticks(self.warpped_frame_chis)
        ax.set_yticklabels(self.warpped_frame_chis)

    def get_theta_chi_warp(
        self,
        dtth,
        dchi,
        tth_start=None,
        tth_end=None,
        chi_start=None,
        chi_end=None,
        use_effective_mask=True,
    ):
        dtth = dtth
        dchi = dchi

        tth_start = self.tths.min() if tth_start is None else tth_start
        tth_end = self.tths.max() if tth_end is None else tth_end

        chi_start = self.chis.min() if chi_start is None else chi_start
        chi_end = self.chis.max() if chi_end is None else chi_end

        tths_values = np.arange(tth_start, tth_end + dtth, dtth)
        # tths_idx = np.arange(len(tths_values))

        chis_values = np.arange(chi_start, chi_end + dchi, dchi)
        # chis_idx = np.arange(len(chis_values))

        # chis_idx_grid, tths_idx_grid = np.meshgrid(chis_idx, tths_idx)
        # chis_values_grid, tths_values_grid = np.meshgrid(chis_values, tths_values)

        x = self.chis.ravel()
        y = self.tths.ravel()
        values = self.frame.data.ravel()

        self.warpped_frame = binned_statistic_2d(
            x=x,
            y=y,
            values=values,
            statistic="sum",
            bins=[
                chis_values,
                tths_values,
            ],
        ).statistic
        if not use_effective_mask:
            self.warpped_effective_mask = np.ones_like(self.warpped_frame, dtype=bool)
        else:
            # mask_rr, mask_cc = self.get_effective_mask_edge()
            # mask_x = self.chis[mask_rr, mask_cc]
            # mask_y = self.tths[mask_rr, mask_cc]
            # mask_w_i = (mask_x - chi_start) // dchi
            # mask_w_j = (mask_y - tth_start) // dtth
            # polygon = np.stack((mask_w_i, mask_w_j), axis=1).astype(int)
            # n = len(polygon) // 50
            # self.warpped_effective_mask = polygon2mask(self.warpped_frame.shape, polygon[::n])

            mask = self.get_effective_mask().ravel()
            self.warpped_effective_mask = (
                binned_statistic_2d(
                    x=x[mask],
                    y=y[mask],
                    values=mask[mask],
                    statistic="sum",
                    bins=[
                        chis_values,
                        tths_values,
                    ],
                ).statistic
                > 0.5
            )

        if use_effective_mask:
            self.warpped_frame *= self.warpped_effective_mask

        self.warpped_frame_tths = tths_values[:-1]
        self.warpped_frame_chis = chis_values[:-1]

        return (
            self.warpped_frame,
            self.warpped_frame_tths,
            self.warpped_frame_chis,
            self.warpped_effective_mask,
        )

    def suggest_integral_params(self):
        tth = np.min(np.abs(self.chis[0:-1, :] - self.chis[1:, :]))
        chi = np.min(np.abs(self.tths[:, 0:-1] - self.tths[:, 1:]))

        return {"tth": (tth * 2, tth * 10), "chi": (chi * 2, tth * 10)}

    def integral_frame(
        self,
        dtth,
        dchi=None,
        tth_start=None,
        tth_end=None,
        chi_start=None,
        chi_end=None,
        use_effective_mask=True,
    ):
        dchi = dtth if dchi is None else dchi
        warpped_frame, frame_tths, frame_chis, frame_mask = self.get_theta_chi_warp(
            dtth,
            dchi,
            tth_start=tth_start,
            tth_end=tth_end,
            chi_start=chi_start,
            chi_end=chi_end,
            use_effective_mask=use_effective_mask,
        )

        integration = warpped_frame.sum(axis=0)

        self.spec = XRDSpectrum(ws=frame_tths, spec=integration)
        return self.spec


class MergeFrames:
    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames
        assert np.all(
            [
                frames[i].detector_spec == frames[i + 1].detector_spec
                for i in range(len(frames) - 1)
            ]
        )
        self.detector_spec = frames[0].detector_spec

    def suggest_integral_params(self):
        params = []
        for frame in self.frames:
            params.append(frame.suggest_integral_params())
        param = {}
        keys = params[0].keys()
        for k in keys:
            param[k] = tuple(np.array([p[k] for p in params]).max(axis=0))

        return param

    def get_theta_chi_warp(
        self,
        dtth,
        dchi,
        tth_start=None,
        tth_end=None,
        chi_start=None,
        chi_end=None,
        use_effective_mask=True,
    ):

        tths_edges = np.arange(tth_start, tth_end + dtth, dtth)
        tths_values = tths_edges[:]
        tths_idx = np.arange(len(tths_values), dtype=int)
        chis_edges = np.arange(chi_start, chi_end + dchi, dchi)
        chis_values = chis_edges[:]
        chis_idx = np.arange(len(chis_values), dtype=int)

        merged_warpped_frame = np.zeros(
            (len(chis_values), len(tths_values)), dtype=float
        )
        merged_warpped_frame_mask = np.zeros(
            (len(chis_values), len(tths_values)), dtype=bool
        )
        merged_weights = np.zeros_like(merged_warpped_frame)

        frames_tth_range = []
        frames_idx_range = []

        for frame in self.frames:
            tth_min, tth_max = frame.tths.min(), frame.tths.max()
            m = (tths_values >= tth_min) & (tths_values < tth_max)

            f_tth_idxs = tths_idx[m][[0, -1]]
            f_tth_range = tths_values[m][[0, -1]]
            f_tth_idxs[-1] = f_tth_idxs[-1]

            frames_tth_range.append(f_tth_range)
            frames_idx_range.append(f_tth_idxs)

        # frames_tth_range[-1][-1] += dtth
        # frames_idx_range[-1][-1] += 2

        warpped_frames = []
        for i, frame in enumerate(self.frames):
            warpped_frame, frame_tths, frame_chis, frame_mask = (
                frame.get_theta_chi_warp(
                    dtth,
                    dchi,
                    tth_start=frames_tth_range[i][0],
                    tth_end=frames_tth_range[i][1],
                    chi_start=chis_values[0],
                    chi_end=chis_values[-1],
                    use_effective_mask=use_effective_mask,
                )
            )

            # print("data tths", frame_tths[[0, -1]], frame_tths.shape, warpped_frame.shape)
            # print("this chis", frames_tth_range[i], frames_idx_range[i])

            # print("data tths", frame_chis[[0, -1]], frame_chis.shape, warpped_frame.shape)
            # print("this chis", chis_values[[0, -1]], chis_values.shape)

            try:
                merged_warpped_frame[:, slice(*frames_idx_range[i])] += warpped_frame
            except Exception as e:
                print(f"merge frame i:{i}")
                print("merged_warpped_frame.shape", merged_warpped_frame.shape)
                print(
                    "data tths",
                    frame_tths[[0, -1]],
                    "frame_tths.shape",
                    frame_tths.shape,
                    "warpped_frame.shape",
                    warpped_frame.shape[1],
                )
                print(
                    "this tths",
                    frames_tth_range[i],
                    "frames_idx_range",
                    frames_idx_range[i],
                )

                print(
                    "data chis",
                    frame_chis[[0, -1]],
                    "frame_chis.shape",
                    frame_chis.shape,
                    "warpped_frame.shape",
                    warpped_frame.shape[0],
                )
                print(
                    "this chis",
                    chis_values[[0, -1]],
                    "chis_values.shape",
                    chis_values.shape,
                )

                raise e
            merged_warpped_frame_mask[:, slice(*frames_idx_range[i])] = np.logical_or(
                merged_warpped_frame_mask[:, slice(*frames_idx_range[i])], frame_mask
            )
            merged_weights[:, slice(*frames_idx_range[i])] += frame_mask

        # merged_weights[~merged_warpped_frame_mask] = 1
        merged_weights = np.clip(merged_weights, 1, None)

        self.raw_warpped_frame = merged_warpped_frame
        merged_warpped_frame = merged_warpped_frame.astype(float) / merged_weights

        self.warpped_frame = merged_warpped_frame
        self.warpped_frame_tths = tths_values
        self.warpped_frame_chis = chis_values
        self.warpped_effective_mask = merged_warpped_frame_mask
        self.warpped_frame_overlap = merged_weights

        return (
            self.warpped_frame,
            self.warpped_frame_tths,
            self.warpped_frame_chis,
            self.warpped_effective_mask,
        )

    def integral_frames(
        self,
        dtth,
        dchi=None,
        tth_start=None,
        tth_end=None,
        chi_start=None,
        chi_end=None,
        use_effective_mask=True,
    ):
        dchi = dtth if dchi is None else dchi
        warpped_frame, frame_tths, frame_chis, frame_mask = self.get_theta_chi_warp(
            dtth,
            dchi,
            tth_start=tth_start,
            tth_end=tth_end,
            chi_start=chi_start,
            chi_end=chi_end,
            use_effective_mask=use_effective_mask,
        )

        if use_effective_mask:
            warpped_frame = warpped_frame * frame_mask

        integration = warpped_frame.sum(axis=0)
        self.spec = XRDSpectrum(ws=frame_tths, spec=integration)
        return self.spec


@dataclass
class Spectrum:
    """
    The Spectrum class is built to abstractalize all spectral data.
    It consists of two fields:
        spec : the intensity of the spectrum stored in an array form
        ws : the location of the intensity stored in an array form
    """

    spec: np.ndarray
    ws: np.ndarray
    spec_label: str = "y"
    ws_label: str = "x"
    spec_unit: str = "a.u."
    ws_unit: str = "a.u."

    def copy(self):
        """
        copy the spectrum object
        """

        return self._update_spectrum(self.spec[...], self.ws[...], inplace=False)

    def _update_spectrum(self, spec, ws, inplace=False, **kargs):
        """
        do an update in the current spectrum's field or create a
        new spectrum and do update over that object.
        """

        if inplace:
            self.spec = spec
            self.ws = ws
            spectrum = self
        else:
            # copy rest of the fields
            stored = {
                field.name: getattr(self, field.name)
                for field in fields(self)
                if field.name not in ["spec", "ws"]
            }
            spectrum = self.__class__(spec, ws, **stored)

        for k, v in kargs.items():
            setattr(spectrum, k, v)

        return spectrum

    def crop(self, sw, ew, inplace=False):
        mask = np.logical_and(self.ws >= sw, self.ws <= ew)
        return self._update_spectrum(self.spec[mask], self.ws[mask], inplace=inplace)

    def interpolate(
        self, ws, fill_value=0, kind="linear", assume_sorted=True, inplace=False
    ):
        """
        Perform interplaction on the spectrum given a new location ws

        Arguments:
            ws : a new location to interpolation on
            fill_value : the constant to fill when location is outside the range of the spectrum
            kind : interpolation method. Could be 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
            assume_sorted : assume x is monotonically increasing value
            inplace : update the current spectrum or create a new spectrum

        Return: Spectrum
        """

        f = interp1d(
            self.ws,
            self.spec,
            bounds_error=False,
            fill_value=fill_value,
            kind=kind,
            assume_sorted=assume_sorted,
        )
        new_ws = ws
        new_spec = f(new_ws)

        return self._update_spectrum(spec=new_spec, ws=new_ws, inplace=inplace)

    def integral(self, sw=None, ew=None):
        sw = self.ws.min() if sw is None else sw
        ew = self.ws.max() if ew is None else ew
        mask = np.logical_and(self.ws >= sw, self.ws <= ew)
        return float(np.trapz(self.spec[mask], self.ws[mask]))

    def normalize(self, min_v=None, max_v=None, inplace=False):
        """
        normalize the spectrum by min, max value
        if min max is not given then it would be computed from the given spectrum

        Arguments:
            min_v : minimum value
            max_v : maximum value
            inplace : if true update current object else return a new object
        """

        _min = self.spec.min() if min_v is None else min_v
        _max = self.spec.max() if max_v is None else max_v

        nspec = (self.spec - _min) / (_max - _min + 1e-5)

        return self._update_spectrum(nspec, self.ws, inplace=inplace)

    def denormalize(self, min_v, max_v, inplace=False):
        """
        Transform the normalized from into the ususal form

        Args:
            min_v (float): minimum value
            max_v ([float): maximum value
            inplace (bool, optional): if true, update the current object else reuturn a new object. Defaults to True.
        """
        nspec = self.spec * (max_v - min_v) + min_v
        return self._update_spectrum(nspec, self.ws, inplace=inplace)

    def scale(self, scalor, inplace=False):
        """
        Scale the signal intensity by the scalor value

        Arguments:
            scalor : scalor value
            inplace : if true update current object else return new object
        """

        return self._update_spectrum(self.spec * scalor, self.ws, inplace=inplace)

    def savgol(
        self,
        window_length=15,
        polyorder=2,
        deriv=0,
        delta=1,
        mode="wrap",
        cval=0,
        inplace=False,
    ):
        """
        Savitzky-Golay filter for noise reduction. Parameters see scipy.signal.savgol_filter

        Arguments:
            window_length : The length of the filter window (i.e., the number of coefficients). Must be odd
            polyorder : The order of the polynomial used to fit the samples. polyorder must be less than window_length.
            deriv : The order of the derivative to compute. This must be a nonnegative integer.
              The default is 0, which means to filter the data without differentiating.
            delta: The spacing of the samples to which the filter will be applied. This is only used if deriv > 0. Default is 1.0.
            mode : Must be ‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’.
                This determines the type of extension to use for the padded signal to which the filter is applied.
                When mode is ‘constant’, the padding value is given by cval. See the Notes for more details on ‘mirror’,
                 ‘constant’, ‘wrap’, and ‘nearest’. When the ‘interp’ mode is selected (the default), no extension is used.
                Instead, a degree polyorder polynomial is fit to the last window_length values of the edges,
                  and this polynomial is used to evaluate the last window_length // 2 output values.

            cval : Value to fill past the edges of the input if mode is ‘constant’. Default is 0.0.

        """
        filtered = savgol_filter(
            self.spec,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            mode=mode,
            cval=0,  # if mode is not constant then whatever
        )
        return self._update_spectrum(filtered, self.ws, inplace=inplace)

    def smooth(self, sigma=1, inplace=False, **kargs):
        nspec = gaussian_filter1d(self.spec, sigma=sigma, **kargs)
        return self._update_spectrum(nspec, self.ws, inplace=inplace)

    def clip(self, clip_min, clip_max, inplace=False):
        return self._update_spectrum(
            np.clip(self.spec, clip_min, clip_max), self.ws, inplace=inplace
        )

    def remove_background_rolling_ball(
        self, radius=100, kernel=None, nansafe=False, num_threads=None, inplace=False
    ):
        """
        Use the rolling ball algorithm to substract any humps in the spectrum
        See skimage.restoration.rolling_ball for more details
        https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rolling_ball.html

        Arguments:
            radius (float) : Radius of a ball shaped kernel to be rolled/translated in the image. Used if kernel = None.
            kernel (ndarray) : The kernel to be rolled/translated in the image. It must have the same number of dimensions as image. Kernel is filled with the intensity of the kernel at that position.
            nansafe (bool): If False (default) assumes that none of the values in image are np.nan, and uses a faster implementation.
            num_threads (int): The maximum number of threads to use. If None use the OpenMP default value; typically equal to the maximum number of virtual cores. Note: This is an upper limit to the number of threads. The exact number is determined by the system’s OpenMP library.

            inplace (bool, optional): if true, update the current object else reuturn a new object. Defaults to True.
        """

        background = restoration.rolling_ball(
            self.spec,
            radius=radius,
            kernel=kernel,
            nansafe=nansafe,
            num_threads=num_threads,
        )
        new_spec = self.spec - background
        return self._update_spectrum(
            new_spec, self.ws, inplace=inplace, background=background
        )

    # Backgroun subtraction
    def remove_background_linear(self, n=5, split=[], inplace=False):
        """
        Assuming the background error is in linear form.
        Fit a linear line from n data points at the beginning and the end of the spectrum.
        Subtract the spacetrum by the fitted linear intensity.

        Arguments:
            n (): number of entries from front and tail to be consider
            split (): position of splitting points
            inplace (bool, optional): if true, update the current object else reuturn a new object. Defaults to True.
        """
        x = self.ws
        y = self.spec

        def _remove_background(x, y):
            if n > 1:
                X = np.concatenate((x[0:n], x[-n:]))
                Y = np.concatenate((y[0:n], y[-n:]))
            else:
                X = np.array([x[0], x[-1]])
                Y = np.array([y[0], y[-1]])
            X = np.stack((X, np.ones_like(X)), axis=1)
            Y = Y.T
            # pdb.set_trace()
            A = np.linalg.inv(X.T @ X) @ (X.T @ Y)
            # pdb.set_trace()
            pX = np.stack((x, np.ones_like(x)), axis=1)
            # pdb.set_trace()
            background = pX @ A
            nspec = y - background
            # pdb.set_trace()
            return nspec, background

        if split:
            ys = []
            bgs = []
            split = [np.min(x)] + split + [np.max(x) + 1e-10]
            for i in range(len(split) - 1):
                mask = np.logical_and(x >= split[i], x < split[i + 1])
                # pdb.set_trace()
                ny, bg = _remove_background(x[mask], y[mask])
                ys.append(ny)
                bgs.append(bg)

            new_spec, background = np.concatenate(ys), np.concatenate(bgs)
        else:
            new_spec, background = _remove_background(x, y)

        return self._update_spectrum(
            new_spec, self.ws, background=background, inplace=inplace
        )

    def filling_flat(self, trunc=0.99, inplace=False):
        """
        Filling truncated area with quadratic spline. Return a
        new PseudoLaueCircleSpectrum if there are area to fill
        else return the original object

        Arguments:
            trunc : maximum value where signal is truncated
        """

        # we fill it by the quadratic curve
        ws, spec = self.ws, self.spec
        smask = spec <= trunc
        if smask.sum() < len(spec) and smask.sum() > 2:
            f = interp1d(ws[smask], spec[smask], kind="quadratic")
            spec = f(ws)

        if inplace:
            self.spec = spec
            return self
        else:
            return Spectrum(spec, self.ws)

    def fit_spectrum_peaks(
        self, height=0.001, threshold=0.001, prominence=0.10, **peak_args
    ):
        """
        Find Peaks over the list of spectrum.

        Arguments:
            height : minimum height of the peak, see find_peaks ref to more detail
            thres : minimum vertical distance to its neighbor peak, see find_peaks ref to more detail
            prominence : peak prominence, see find_peaks ref to more detail
        """
        peaks, peaks_info = find_peaks(
            self.spec,
            height=height,
            threshold=threshold,
            prominence=prominence,
            **peak_args,
        )

        self.peaks = peaks
        self.peaks_info = peaks_info

        return peaks, peaks_info

    def plot_spectrum(
        self,
        ax=None,
        peaks=None,
        peakgroups=None,
        offset=0,
        peak_offset=0,
        showlegend=False,
        linecolor=None,
        linewidth=1,
        use_log_scale=False,
        spectrum_name=None,
        **fig_kargs,
    ):
        """
        Plot the spectrum using matplotlib
        Arguments:
            ax : the matplotlib Axes to plot onto, if None then a new figure is created
            peaks : the peaks index in array form
            peakgroups : a group index of peak's index telling how peak index is group togethered
            offset : a offset that life the spectrum line
            peak_offset : a offset that lift the peak symbol away from the spectrum line
            showlegend : show legend
        Return
            fig : matplotlib Figure
            ax : matplotlib Axes
        """

        # peaks, peaksinfo = peaks
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        peak_offset_arr = np.zeros_like(peaks, dtype=float)

        ax.plot(
            self.ws,
            self.spec + offset,
            c=linecolor,
            linewidth=linewidth,
            label=spectrum_name,
        )

        if peaks is not None:
            if peakgroups is not None:
                for g, dist in peakgroups[:-1]:
                    ax.plot(
                        self.ws[peaks[g]],
                        self.spec[peaks[g]] + offset + peak_offset_arr[g],
                        "x",
                        label=f"dist={dist:.1f}",
                    )
                    peak_offset_arr[g] += peak_offset

                g, _ = peakgroups[-1]
                ax.plot(
                    self.ws[peaks[g]],
                    self.spec[peaks[g]] + offset + offset + peak_offset_arr[g],
                    "o",
                )
            else:
                ax.plot(self.ws[peaks], self.spec[peaks] + offset + peak_offset, "x")

        if showlegend:
            ax.legend()

        if use_log_scale:
            ax.set_yscale("log")
        ax.set_xlabel(f"{self.ws_label} ({self.ws_unit})")
        ax.set_ylabel(f"{self.spec_label} ({self.spec_unit})")

        return fig, ax

    def save(self, path, name="spectrum"):
        _exclude = ["sm"]

        path = Path(path)
        temps = {k: v for k, v in self.__dict__.items() if k in _exclude}

        for k in _exclude:
            setattr(self, k, None)

        save_pickle(self, path / f"{name}.pkl")

        for k, v in temps.items():
            setattr(self, k, v)

    @classmethod
    def load(cls, path, name="spectrum"):
        self = load_pickle(path / f"{name}.pkl")

        return self


@dataclass
class XRDSpectrum(Spectrum):
    spec_label: str = "Intensity"
    ws_label: str = "2θ"
    spec_unit: str = "count"
    ws_unit: str = "°"

    wavelengths: Optional[Tuple] = None
    frame_angles: Optional[Dict[str, float]] = None
    integration_range: Optional[Dict[str, Tuple]] = None
    integration_method: Optional[str] = None
    name: Optional[str] = None
    metas: Optional[Dict] = None

    def convert_ws(self, axis_type: str, wavelength: float = None):
        """
        a convertion between 2θ and q. Here we assume wavelengths is in Å

        Arguments:
            axis_type (str) : could be "tt" or "q"
        """
        ws, ws_label, ws_unit = convert_ws(
            ws=self.ws,
            ws_label=self.ws_label,
            axis_type=axis_type,
            wavelength=wavelength,
        )

        # if axis_type == "q" and self.ws_label=="2θ":
        #     ws = np.sin(np.deg2rad(self.ws/2)) * 4 * np.pi / self.wavelengths
        #     ws_label = "2θ"
        #     ws_unit = "°"

        # elif axis_type == "tt" and self.ws_label=="Q":
        #     ws = np.arcsin(self.ws / (4 * np.pi / self.wavelengths) ) * 2
        #     ws_label = "Q"
        #     ws_unit = "1/Å"
        # else:
        #     raise ValueError("Argument axis_type must be one of ['tt', 'q']")

        return self._update_spectrum(
            spec=self.spec, ws=ws, inplace=False, ws_unit=ws_unit, ws_label=ws_label
        )

    @classmethod
    def from_plt_file(cls, path):
        pltf = PltFile(path)
        return cls(
            spec=pltf.y,
            ws=pltf.x,
            wavelengths=pltf.wavelengths,
            frame_angles=pltf.frame_angles,
            integration_range=pltf.integration_range,
            integration_method=pltf.integration_method,
            name=pltf.name,
        )

    @classmethod
    def from_xy_file(cls, path, **kargs):
        xyf = XYFile(path)

        return cls(spec=xyf.y, ws=xyf.x, name=path.stem, metas=xyf.headers, **kargs)

    @classmethod
    def from_npy_file(cls, path, npy_kargs={}, **kargs):
        array = np.load(path, **npy_kargs)

        return cls(spec=array[:, 1], ws=array[:, 0], name=path.stem**kargs)

    @classmethod
    def from_x00_file(cls, path, **kargs):
        x00_file = X00File(path)
        return cls(
            spec=x00_file.y,
            ws=x00_file.x,
            name=path.stem,
            metas=x00_file.metas,
            **kargs,
        )

    def save(self, path, name="xrd"):
        super().save(path, name)

    @classmethod
    def load(cls, path, name="xrd"):
        return super(XRDSpectrum, cls).load(path, name)
