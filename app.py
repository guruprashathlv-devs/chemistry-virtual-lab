import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Consistent chart styling across the lab; keep plots legible on desktop and mobile.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d9e4ec",
    "axes.labelcolor": "#334e68",
    "axes.titlecolor": "#102a43",
    "xtick.color": "#486581",
    "ytick.color": "#486581",
    "grid.color": "#d9e4ec",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "legend.frameon": False,
})

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Draw,
    AllChem,
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors
)

# 3D Visualization
import py3Dmol
import streamlit.components.v1 as components

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    Preformatted
)
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import tempfile
import os
import json
import math
from datetime import datetime, timezone
import uuid

from storage import (
    make_student_key, save_activity, save_quiz_attempt,
    get_activities, get_quiz_attempts, get_progress_summary,
)
from ai_tutor import is_configured as ai_is_configured, ask_tutor
from chemistry_core import (
    strong_acid_strong_base_ph, first_order_concentration, first_order_half_life,
    nernst_potential, beer_lambert_absorbance, beer_lambert_concentration,
    arrhenius_rate_constant,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --lab-ink:#102a43; --lab-muted:#627d98; --lab-teal:#087e8b; --lab-blue:#2563eb; --lab-line:#d9e4ec; --lab-surface:#f5f8fb; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f7fafc 0%,#ffffff 260px); color:var(--lab-ink); }
.block-container { max-width:1440px; padding-top:1.5rem; padding-bottom:3rem; }
.main-title { text-align:center; font-family:'Manrope',sans-serif; font-size:clamp(1.9rem,3vw,2.8rem); line-height:1.15; font-weight:800; letter-spacing:-.04em; color:var(--lab-ink); padding:.8rem 0 .5rem; }
.section-title { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; letter-spacing:-.02em; color:var(--lab-ink); }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.025em; color:var(--lab-ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#edf5f8 0%,#f8fbfd 100%); border-right:1px solid var(--lab-line); }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--lab-ink); }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--lab-line); border-radius:16px; padding:14px 16px; box-shadow:0 2px 8px rgba(16,42,67,.035); }
[data-testid="stMetricLabel"] { color:var(--lab-muted); font-weight:600; }
[data-testid="stMetricValue"] { color:var(--lab-ink); font-family:'Manrope',sans-serif; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button { border-radius:10px; font-weight:700; min-height:2.65rem; transition:transform .15s ease, box-shadow .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 5px 14px rgba(8,126,139,.14); }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] { background:linear-gradient(135deg,#087e8b,#2563eb); border:0; }
[data-testid="stTabs"] button { font-weight:650; }
[data-testid="stTabs"] button[aria-selected="true"] { color:var(--lab-teal); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--lab-line); border-radius:12px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stPlotlyChart"], [data-testid="stPyplot"] { background:#fff; border:1px solid var(--lab-line); border-radius:14px; padding:8px; }
hr { border-color:var(--lab-line); }
.small-muted { color:var(--lab-muted); font-size:.9rem; }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; scroll-behavior:auto !important; } }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

# ============================================================
# SHARED MOLECULE INPUT STATE
# ============================================================

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CCO"

if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Ethanol"

def load_shared_example():
    """Load the selected example into the shared SMILES field."""
    st.session_state.active_smiles = EXAMPLE_MOLECULES[
        st.session_state.selected_example
    ]

def get_molecule(smiles):
    """Convert SMILES into an RDKit molecule."""

    if not smiles or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol

    except Exception:
        return None


# ============================================================
# CALCULATE SIGMA AND PI BONDS
# ============================================================

def calculate_bond_information(mol):
    """
    Calculate:
    - Heavy-atom bonds
    - Sigma (σ) bonds, including bonds to implicit hydrogens
    - Pi (π) bonds
    - Total bond order count (σ + π)
    """

    # Heavy-atom graph bonds (RDKit default molecule representation)
    heavy_atom_bonds = mol.GetNumBonds()

    # Count π bonds using a Kekulé representation for aromatic systems
    kekule_mol = Chem.Mol(mol)

    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        # If Kekulization is not possible, continue with the available form
        pass

    pi_bonds = 0

    for bond in kekule_mol.GetBonds():
        bond_type = bond.GetBondType()

        if bond_type == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond_type == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond_type == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            # Fallback for aromatic bonds if Kekulization did not clear them
            pi_bonds += 0.5

    # Add implicit hydrogens. Every atom-to-atom connection contains one σ bond.
    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()

    total_bonds = sigma_bonds + pi_bonds

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": total_bonds,
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
    }


# ============================================================
# CALCULATE COMPLETE MOLECULAR PROPERTIES
# ============================================================

def calculate_properties(mol):

    # Calculate bond information
    bond_info = calculate_bond_information(mol)

    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)

    properties = {

        "Molecular Formula": molecular_formula,

        "Molecular Weight": round(
            Descriptors.MolWt(mol),
            3
        ),

        "Exact Molecular Weight": round(
            Descriptors.ExactMolWt(mol),
            4
        ),

        "LogP": round(
            Crippen.MolLogP(mol),
            3
        ),

        "TPSA": round(
            rdMolDescriptors.CalcTPSA(mol),
            3
        ),

        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),

        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),

        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),

        "Ring Count": Lipinski.RingCount(mol),

        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),

        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),

        "Number of Atoms": mol.GetNumAtoms(),

        "Heavy Atoms": mol.GetNumHeavyAtoms(),

        # Bond information
        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],

        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],

        "Pi (π) Bonds": bond_info["Pi Bonds"],

        "Total Bonds (σ + π)": bond_info["Total Bonds"],

        "Fraction Csp3": round(
            rdMolDescriptors.CalcFractionCSP3(mol),
            3
        ),

        "Molar Refractivity": round(
            Crippen.MolMR(mol),
            3
        )
    }

    return properties

# ============================================================
# GENERATE 2D IMAGE
# ============================================================

def generate_2d_image(mol):

    image = Draw.MolToImage(
        mol,
        size=(700, 500)
    )

    return image


# ============================================================
# GENERATE 3D MOLECULE
# ============================================================

def generate_3d_molecule(mol):

    try:

        mol3d = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()

        params.randomSeed = 42

        status = AllChem.EmbedMolecule(
            mol3d,
            params
        )

        if status != 0:
            return None

        try:

            AllChem.MMFFOptimizeMolecule(
                mol3d
            )

        except Exception:

            try:

                AllChem.UFFOptimizeMolecule(
                    mol3d
                )

            except Exception:
                pass

        return mol3d

    except Exception:

        return None


# ============================================================
# DISPLAY 3D STRUCTURE
# ============================================================

def display_3d_structure(mol3d, style, show_surface=False):

    mol_block = Chem.MolToMolBlock(mol3d)

    view = py3Dmol.view(
        width=900,
        height=600
    )

    view.addModel(
        mol_block,
        "mol"
    )

    # --------------------------------------------------------
    # BALL AND STICK
    # --------------------------------------------------------

    if style == "Ball and Stick":

        view.setStyle({

            "stick": {
                "radius": 0.15
            },

            "sphere": {
                "scale": 0.30
            }

        })

    # --------------------------------------------------------
    # STICK
    # --------------------------------------------------------

    elif style == "Stick":

        view.setStyle({

            "stick": {
                "radius": 0.20
            }

        })

    # --------------------------------------------------------
    # SPACE FILLING
    # --------------------------------------------------------

    elif style == "Space Filling":

        view.setStyle({

            "sphere": {
                "scale": 1.0
            }

        })

    # --------------------------------------------------------
    # WIREFRAME
    # --------------------------------------------------------

    elif style == "Wireframe":

        view.setStyle({

            "line": {
                "linewidth": 2
            }

        })

    # --------------------------------------------------------
    # CARTOON STYLE
    # --------------------------------------------------------

    elif style == "Cartoon":

        # Cartoon is mainly useful for biomolecules,
        # but this provides a molecular representation.

        view.setStyle({

            "stick": {},

            "sphere": {
                "scale": 0.20
            }

        })

    # --------------------------------------------------------
    # OPTIONAL SURFACE
    # --------------------------------------------------------

    if show_surface:

        view.addSurface(
            py3Dmol.VDW,
            {
                "opacity": 0.7
            }
        )

    view.setBackgroundColor("white")

    view.zoomTo()

    html = view._make_html()

    components.html(
        html,
        height=620
    )


# ============================================================
# CREATE MOLECULAR PDF REPORT
# ============================================================

def generate_molecule_pdf(
    input_smiles,
    mol,
    properties
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=20,
        alignment=TA_CENTER,
        textColor=HexColor("#17365D")
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=HexColor("#17365D")
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )

    story = []

    story.append(
        Paragraph(
            "MOLECULAR PROPERTIES REPORT",
            title_style
        )
    )

    story.append(
        Spacer(1, 0.3 * inch)
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Molecular Information",
            heading_style
        )
    )

    canonical_smiles = Chem.MolToSmiles(mol)

    basic_data = [

        ["Property", "Value"],

        ["Input SMILES", input_smiles],

        ["Canonical SMILES", canonical_smiles],

        [
            "Molecular Formula",
            properties["Molecular Formula"]
        ]
    ]

    basic_table = Table(
        basic_data,
        colWidths=[2 * inch, 4 * inch]
    )

    basic_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#17365D")),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("PADDING", (0, 0), (-1, -1), 7)

        ])
    )

    story.append(basic_table)

    story.append(
        Spacer(1, 0.3 * inch)
    )

    # --------------------------------------------------------
    # PROPERTY TABLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Molecular Properties",
            heading_style
        )
    )

    property_data = [
        ["Property", "Value"]
    ]

    for key, value in properties.items():

        property_data.append(
            [key, str(value)]
        )

    property_table = Table(
        property_data,
        colWidths=[3 * inch, 3 * inch],
        repeatRows=1
    )

    property_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("PADDING", (0, 0), (-1, -1), 6)

        ])
    )

    story.append(property_table)

    story.append(
        Spacer(1, 0.3 * inch)
    )

    story.append(
        Paragraph(
            "3. Interpretation",
            heading_style
        )
    )

    interpretation = """
    This report was generated automatically using the
    Cheminformatics Virtual Laboratory. The molecular properties
    were calculated from the chemical structure represented by
    the submitted SMILES notation using the RDKit cheminformatics
    toolkit.
    """

    story.append(
        Paragraph(
            interpretation,
            body_style
        )
    )

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    doc.build(story)

    pdf_data = buffer.getvalue()

    buffer.close()

    return pdf_data


# ============================================================
# REPORT IMAGE HELPERS
# ============================================================

def save_2d_structure_png(mol):
    """Create a temporary PNG for inclusion in PDF reports."""
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name


def save_3d_structure_png(mol):
    """
    Create a high-clarity ball-and-stick image for the PDF report.

    The RDKit 3D coordinates are projected onto the best viewing plane so
    elongated molecules fill the page instead of appearing as a tiny object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    n_atoms = mol3d.GetNumAtoms()

    # Collect 3D coordinates.
    coords = np.array([
        [
            conf.GetAtomPosition(i).x,
            conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z,
        ]
        for i in range(n_atoms)
    ], dtype=float)

    # Principal-component projection chooses the clearest overall view.
    centered = coords - coords.mean(axis=0)
    if n_atoms >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    # CPK-style colours and readable atom sizes.
    cpk = {
        "H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF",
        "O": "#E53935", "F": "#65C466", "P": "#F39C12",
        "S": "#F4D03F", "Cl": "#27AE60", "Br": "#8E3B2F",
        "I": "#8E44AD",
    }
    sizes = {
        "H": 70, "C": 240, "N": 270, "O": 285, "F": 260,
        "P": 300, "S": 310, "Cl": 310, "Br": 330, "I": 350,
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")

    # Draw bonds first. Multiple bonds are shown with parallel lines.
    for bond in mol3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]

        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue

        # Perpendicular direction for double/triple bond separation.
        px, py = -dy / length, dx / length
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            offsets = (-0.035, 0.035)
        elif btype == Chem.BondType.TRIPLE:
            offsets = (-0.055, 0.0, 0.055)
        else:
            offsets = (0.0,)

        for off in offsets:
            ax.plot(
                [x1 + px * off, x2 + px * off],
                [y1 + py * off, y2 + py * off],
                color="#707070", linewidth=3.0,
                solid_capstyle="round", zorder=1
            )

    # Draw atoms over bonds.
    for i, atom in enumerate(mol3d.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(
            x, y,
            s=sizes.get(symbol, 250),
            c=cpk.get(symbol, "#A0A0A0"),
            edgecolors="#222222",
            linewidths=1.0,
            zorder=3
        )

        # Label important/non-carbon hetero atoms only; this keeps the model clean.
        if symbol not in ("H", "C"):
            ax.text(
                x, y, symbol,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="black", zorder=4
            )

    # Tight, equal-aspect limits make the molecule large and clear.
    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    pad = 0.22 * max(width, height)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("3D Molecular Structure (Ball-and-Stick)", fontsize=18, fontweight="bold", pad=18)

    fig.tight_layout(pad=1.2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(
        tmp.name,
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.12
    )
    plt.close(fig)
    return tmp.name


# ============================================================
# PROPERTY INTERPRETATION HELPERS
# ============================================================

def get_property_interpretation(property_name, value, properties):
    """Return a student-friendly interpretation for each reported property."""

    if property_name == "Molecular Formula":
        return (
            "Shows the elemental composition of the molecule and is useful "
            "for understanding its basic chemical composition."
        )

    if property_name == "Molecular Weight":
        return (
            "Represents the average molecular mass. Molecular size and mass "
            "can influence diffusion, formulation and many physicochemical behaviours."
        )

    if property_name == "Exact Molecular Weight":
        return (
            "The monoisotopic mass calculated from exact isotopic masses; "
            "this is especially useful when comparing mass-spectrometry data."
        )

    if property_name == "LogP":
        if value < 0:
            return "A negative value indicates a preference toward the aqueous phase and relatively high hydrophilicity."
        elif value < 1:
            return "Indicates low to moderate lipophilicity, suggesting a reasonable preference for polar environments."
        elif value < 3:
            return "Indicates moderate lipophilicity, often reflecting a balance between hydrophilic and hydrophobic character."
        else:
            return "Indicates relatively high lipophilicity and a stronger preference for non-polar environments."

    if property_name == "TPSA":
        if value < 60:
            return "A relatively low polar surface area, indicating limited overall molecular polarity."
        elif value < 140:
            return "A moderate polar surface area, reflecting the contribution of heteroatoms to molecular polarity."
        else:
            return "A high polar surface area, indicating substantial polarity and strong hydrogen-bonding potential."

    if property_name == "Hydrogen Bond Donors":
        return (
            f"The molecule has {value} donor site(s) capable of donating hydrogen in hydrogen-bond interactions."
        )

    if property_name == "Hydrogen Bond Acceptors":
        return (
            f"The molecule has {value} acceptor site(s) capable of accepting hydrogen in hydrogen-bond interactions."
        )

    if property_name == "Rotatable Bonds":
        if value == 0:
            return "No freely rotatable bonds are detected, suggesting a relatively rigid molecular framework."
        return f"{value} rotatable bond(s) indicate conformational flexibility in the molecular structure."

    if property_name == "Ring Count":
        return f"The molecule contains {value} ring(s), which contribute to its overall shape and structural rigidity."

    if property_name == "Aromatic Rings":
        return f"{value} aromatic ring(s) contribute to aromaticity, planarity and delocalized π-electron character."

    if property_name == "Aliphatic Rings":
        return f"{value} non-aromatic ring(s) contribute to cyclic structure without aromatic π-electron delocalization."

    if property_name == "Number of Atoms":
        return "Counts atoms explicitly present in the RDKit molecular representation; implicit hydrogens are not included here."

    if property_name == "Heavy Atoms":
        return "Counts all non-hydrogen atoms and provides a useful measure of molecular structural size."

    if property_name == "Heavy Atom Bonds":
        return "Counts connections between non-hydrogen atoms in the molecular graph."

    if property_name == "Sigma (σ) Bonds":
        return "Every bonded atom pair contains one σ bond; this value includes bonds involving explicit or added hydrogens."

    if property_name == "Pi (π) Bonds":
        return "π bonds arise from multiple bonds and aromatic systems and are associated with electron delocalization."

    if property_name == "Total Bonds (σ + π)":
        return "Represents the total bond components obtained by adding the calculated σ and π bond counts."

    if property_name == "Fraction Csp3":
        if value < 0.25:
            return "A low sp³ fraction suggests a comparatively planar, unsaturated or aromatic structural character."
        elif value < 0.60:
            return "An intermediate sp³ fraction indicates a balance between saturated and unsaturated structural features."
        return "A high sp³ fraction indicates a more saturated, three-dimensional molecular framework."

    if property_name == "Molar Refractivity":
        return "Relates to molecular volume and electronic polarizability and reflects how the electron cloud responds to an electric field."

    return "This descriptor provides additional quantitative information about the molecular structure and physicochemical behaviour."


def build_key_interpretation_points(properties):
    """Create concise point-wise conclusions for the final report."""
    points = []

    points.append(
        f"<b>• Molecular identity:</b> The submitted structure has molecular formula "
        f"<b>{properties['Molecular Formula']}</b> and molecular weight "
        f"<b>{properties['Molecular Weight']}</b>."
    )

    logp = properties["LogP"]
    if logp < 0:
        lipophilic_text = "more hydrophilic than lipophilic"
    elif logp < 1:
        lipophilic_text = "low in lipophilicity"
    elif logp < 3:
        lipophilic_text = "moderately lipophilic"
    else:
        lipophilic_text = "strongly lipophilic"

    points.append(
        f"<b>• Lipophilicity:</b> LogP = <b>{logp}</b>, indicating that the molecule is "
        f"<b>{lipophilic_text}</b>. This descriptor describes the balance between "
        f"preference for non-polar and polar environments."
    )

    points.append(
        f"<b>• Molecular polarity:</b> TPSA = <b>{properties['TPSA']}</b> Å². "
        f"TPSA reflects the contribution of polar atoms and functional groups to the "
        f"overall surface polarity of the molecule."
    )

    points.append(
        f"<b>• Hydrogen bonding:</b> The structure contains "
        f"<b>{properties['Hydrogen Bond Donors']}</b> hydrogen-bond donor(s) and "
        f"<b>{properties['Hydrogen Bond Acceptors']}</b> acceptor(s), which helps "
        f"describe possible intermolecular interactions."
    )

    points.append(
        f"<b>• Flexibility:</b> {properties['Rotatable Bonds']} rotatable bond(s) "
        f"indicate the degree of conformational freedom available to the molecule."
    )

    points.append(
        f"<b>• Cyclic and aromatic character:</b> The molecule contains "
        f"{properties['Ring Count']} ring(s), including "
        f"{properties['Aromatic Rings']} aromatic ring(s). Aromatic rings are "
        f"associated with delocalized π-electron systems."
    )

    points.append(
        f"<b>• Bonding pattern:</b> The calculated structure contains "
        f"<b>{properties['Sigma (σ) Bonds']}</b> σ bonds and "
        f"<b>{properties['Pi (π) Bonds']}</b> π bonds. This helps distinguish the "
        f"framework-forming σ bonds from multiple-bond and delocalized π bonding."
    )

    points.append(
        f"<b>• Three-dimensional character:</b> Fraction Csp³ = "
        f"<b>{properties['Fraction Csp3']}</b>. This descriptor provides a simple "
        f"indication of the balance between saturated 3D carbon environments and "
        f"planar/unsaturated environments."
    )

    points.append(
        "<b>• Overall conclusion:</b> The calculated descriptors should be interpreted "
        "together with the 2D connectivity and generated 3D geometry to understand "
        "the molecule's composition, bonding, polarity, flexibility and structural character."
    )

    return points


# ============================================================
# GENERATE GENERAL FINAL PROJECT REPORT
# ============================================================

def generate_final_report(student_name, registration_number, input_smiles, mol, properties):
    """Generate a detailed personalized final report for the student's molecule."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ProjectTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=HexColor("#17365D")
    )

    heading_style = ParagraphStyle(
        "ProjectHeading",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=HexColor("#17365D"),
        spaceBefore=8,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ProjectBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY
    )

    small_style = ParagraphStyle(
        "SmallTableText",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9.5
    )

    header_style = ParagraphStyle(
        "HeaderTableText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=9,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    story = []

    story += [
        Spacer(1, 0.25 * inch),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.12 * inch),
        Paragraph("FINAL STUDENT MOLECULAR ANALYSIS REPORT", heading_style),
        Spacer(1, 0.18 * inch)
    ]

    student_data = [
        ["Student Name", student_name or "Not provided"],
        ["Registration Number", registration_number or "Not provided"],
        ["Input SMILES", input_smiles],
        ["Canonical SMILES", Chem.MolToSmiles(mol)],
        ["Molecular Formula", properties["Molecular Formula"]]
    ]

    student_table = Table(student_data, colWidths=[2.0 * inch, 4.7 * inch])
    student_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF0F8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story += [student_table, Spacer(1, 0.22 * inch)]

    # --------------------------------------------------------
    # 1. STRUCTURES
    # --------------------------------------------------------
    story.append(Paragraph("1. Molecular Structures", heading_style))

    img2d = save_2d_structure_png(mol)
    img3d = save_3d_structure_png(mol)

    if img2d and img3d:
        structure_table = Table(
            [[
                Image(img2d, width=3.15 * inch, height=2.25 * inch),
                Image(img3d, width=3.15 * inch, height=2.25 * inch)
            ]],
            colWidths=[3.3 * inch, 3.3 * inch]
        )

        structure_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER")
        ]))

        story += [
            structure_table,
            Spacer(1, 0.08 * inch),
            Paragraph(
                "<b>Figure 1.</b> Left: 2D molecular connectivity. "
                "Right: generated 3D ball-and-stick geometry showing the spatial arrangement of atoms.",
                body_style
            )
        ]

    elif img2d:
        story.append(Image(img2d, width=5.8 * inch, height=4.1 * inch))

    story += [Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 2. COMPLETE PROPERTIES WITH INTERPRETATION COLUMN
    # --------------------------------------------------------
    story.append(Paragraph("2. Complete Molecular Properties and Interpretation", heading_style))

    prop_data = [[
        Paragraph("<b>Property</b>", header_style),
        Paragraph("<b>Value</b>", header_style),
        Paragraph("<b>Interpretation</b>", header_style)
    ]]

    for key, value in properties.items():
        interpretation = get_property_interpretation(key, value, properties)
        prop_data.append([
            Paragraph(str(key), small_style),
            Paragraph(str(value), small_style),
            Paragraph(interpretation, small_style)
        ])

    property_table = Table(
        prop_data,
        colWidths=[1.75 * inch, 1.0 * inch, 3.85 * inch],
        repeatRows=1
    )

    property_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4)
    ]))

    story += [property_table, Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 3. POINT-WISE INTERPRETATION
    # --------------------------------------------------------
    story.append(Paragraph("3. Point-wise Molecular Interpretation", heading_style))

    intro = (
        "The following points summarize the most important structural and "
        "physicochemical features calculated from the student's submitted SMILES."
    )
    story.append(Paragraph(intro, body_style))
    story.append(Spacer(1, 0.08 * inch))

    for point in build_key_interpretation_points(properties):
        story.append(Paragraph(point, body_style))
        story.append(Spacer(1, 0.06 * inch))

    # --------------------------------------------------------
    # 4. LEARNING NOTE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.12 * inch),
        Paragraph("4. Learning Note", heading_style),
        Paragraph(
            "The numerical descriptors in this report are computationally calculated "
            "from the submitted molecular structure. They are useful for molecular "
            "comparison and cheminformatics learning, but experimental measurements "
            "may differ depending on conditions and methodology.",
            body_style
        )
    ]


    # --------------------------------------------------------
    # 5. PYTHON LIBRARIES AND THEIR IMPORTANCE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.12 * inch),
        Paragraph("5. Python Libraries Used and Their Importance", heading_style),
        Paragraph(
            "The virtual laboratory uses the following libraries to process "
            "molecular structures, calculate descriptors, display results and "
            "create this PDF report.",
            body_style
        ),
        Spacer(1, 0.08 * inch)
    ]

    library_rows = [
        [
            Paragraph("<b>Library</b>", header_style),
            Paragraph("<b>Importance in this project</b>", header_style)
        ],
        [
            Paragraph("Streamlit", small_style),
            Paragraph("Builds the interactive web application, input fields, navigation, "
                      "visualization controls and download buttons.", small_style)
        ],
        [
            Paragraph("RDKit (Chem, Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw, AllChem)",
                      small_style),
            Paragraph("Reads SMILES, validates molecular structures, generates 2D drawings "
                      "and 3D coordinates, and calculates molecular formula, molecular weight, "
                      "LogP, TPSA, hydrogen-bond counts, ring counts and other descriptors.",
                      small_style)
        ],
        [
            Paragraph("Pandas", small_style),
            Paragraph("Organizes molecular properties into tables and supports CSV export.",
                      small_style)
        ],
        [
            Paragraph("Matplotlib", small_style),
            Paragraph("Creates a clear 2D projection of the generated 3D molecular geometry "
                      "for inclusion in the PDF report.", small_style)
        ],
        [
            Paragraph("NumPy", small_style),
            Paragraph("Processes 3D atomic coordinates and uses numerical linear algebra "
                      "to select a clear projection of the molecule.", small_style)
        ],
        [
            Paragraph("py3Dmol", small_style),
            Paragraph("Displays interactive 3D molecular models in the web application.",
                      small_style)
        ],
        [
            Paragraph("ReportLab", small_style),
            Paragraph("Creates the downloadable PDF, including student details, structures, "
                      "property tables, interpretations, code examples and library notes.",
                      small_style)
        ],
        [
            Paragraph("Python standard libraries: io, os, tempfile", small_style),
            Paragraph("Manage in-memory PDF data and temporary image files, and clean up "
                      "temporary files after report generation.", small_style)
        ]
    ]

    library_table = Table(
        library_rows,
        colWidths=[2.05 * inch, 4.65 * inch],
        repeatRows=1
    )
    library_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story += [library_table, Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 6. ILLUSTRATIVE PYTHON CODE
    # --------------------------------------------------------
    story.append(Paragraph("6. Python Code: SMILES to Molecular Properties", heading_style))
    story.append(Paragraph(
        "The following educational example shows the core workflow used by this "
        "laboratory. It is a compact example of the main calculation steps, not the "
        "complete Streamlit application source code.",
        body_style
    ))
    story.append(Spacer(1, 0.08 * inch))

    code_style = ParagraphStyle(
        "EducationalCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7,
        leading=8.5,
        leftIndent=5,
        rightIndent=5,
        spaceBefore=4,
        spaceAfter=4
    )

    educational_code = '''from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw
from rdkit.Chem import AllChem

# 1. Convert a SMILES string into a molecular object
smiles = "CCO"  # Example: ethanol
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    raise ValueError("Invalid SMILES notation")

# 2. Calculate selected molecular descriptors
formula = rdMolDescriptors.CalcMolFormula(mol)
molecular_weight = Descriptors.MolWt(mol)
logp = Crippen.MolLogP(mol)
tpsa = rdMolDescriptors.CalcTPSA(mol)
h_bond_donors = Lipinski.NumHDonors(mol)
h_bond_acceptors = Lipinski.NumHAcceptors(mol)

print("Formula:", formula)
print("Molecular weight:", round(molecular_weight, 3), "g/mol")
print("LogP:", round(logp, 3))
print("TPSA:", round(tpsa, 3), "A^2")
print("H-bond donors:", h_bond_donors)
print("H-bond acceptors:", h_bond_acceptors)

# 3. Save a 2D molecular structure image
Draw.MolToFile(mol, "molecule_2d.png", size=(700, 500))

# 4. Generate a 3D conformer (coordinates)
mol_3d = Chem.AddHs(mol)
status = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
if status == 0:
    AllChem.MMFFOptimizeMolecule(mol_3d)
    print("3D coordinates generated successfully")'''

    story.append(Preformatted(educational_code, code_style, maxLineLength=92))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("How to read the code", heading_style))
    code_notes = [
        "<b>SMILES input:</b> A text representation of the molecular connectivity.",
        "<b>Validation:</b> <font name='Courier'>Chem.MolFromSmiles()</font> attempts to parse the input; "
        "an invalid structure returns <font name='Courier'>None</font>.",
        "<b>Descriptors:</b> RDKit functions calculate numerical properties from the parsed molecular graph.",
        "<b>2D drawing:</b> <font name='Courier'>Draw.MolToFile()</font> writes a molecular diagram to an image file.",
        "<b>3D geometry:</b> Explicit hydrogens are added, a conformer is generated, and MMFF optimization "
        "is attempted to refine its geometry."
    ]
    for note in code_notes:
        story.append(Paragraph("• " + note, body_style))
        story.append(Spacer(1, 0.04 * inch))

    # --------------------------------------------------------
    # 7. REPRODUCIBILITY NOTE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.10 * inch),
        Paragraph("7. Reproducibility and Limitations", heading_style),
        Paragraph(
            "This code is an educational illustration of the principal workflow. "
            "The web application contains additional functions for student inputs, "
            "property interpretation, 3D display and PDF formatting. Calculated "
            "descriptors are computational estimates and should be interpreted with "
            "their definitions and, where appropriate, compared with experimental data.",
            body_style
        )
    ]


    doc.build(story)

    pdf_data = buffer.getvalue()
    buffer.close()

    for path in [img2d, img3d]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    return pdf_data



# ============================================================
# GENERATE ONE PDF FOR MULTIPLE MOLECULES
# ============================================================

def generate_multi_final_report(student_name, registration_number, molecule_records):
    """Create one PDF containing results for every valid SMILES supplied."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=38, rightMargin=38,
        topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MultiReportTitle", parent=styles["Title"], fontSize=19,
        leading=23, alignment=TA_CENTER, textColor=HexColor("#17365D")
    )
    heading_style = ParagraphStyle(
        "MultiReportHeading", parent=styles["Heading1"], fontSize=13,
        leading=16, textColor=HexColor("#17365D"), spaceBefore=7, spaceAfter=7
    )
    body_style = ParagraphStyle(
        "MultiReportBody", parent=styles["BodyText"], fontSize=9,
        leading=12, alignment=TA_JUSTIFY
    )
    small_style = ParagraphStyle(
        "MultiReportSmall", parent=styles["BodyText"], fontSize=7.2, leading=8.8
    )
    header_style = ParagraphStyle(
        "MultiReportHeader", parent=styles["BodyText"], fontSize=7.5,
        leading=9, textColor=colors.white, alignment=TA_CENTER
    )
    story = [
        Spacer(1, 0.15 * inch),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.08 * inch),
        Paragraph("MULTI-MOLECULE STUDENT ANALYSIS REPORT", heading_style),
        Paragraph(
            f"Student: {student_name or 'Not provided'} &nbsp;&nbsp;&nbsp; "
            f"Registration Number: {registration_number or 'Not provided'}",
            body_style
        ),
        Paragraph(f"Number of molecules analyzed: {len(molecule_records)}", body_style),
        Spacer(1, 0.15 * inch)
    ]

    temp_images = []
    for index, record in enumerate(molecule_records, start=1):
        input_smiles, mol, properties = record
        if index > 1:
            story.append(PageBreak())

        story.append(Paragraph(f"Molecule {index}: Molecular Information", heading_style))
        basic_data = [
            ["Input SMILES", input_smiles],
            ["Canonical SMILES", Chem.MolToSmiles(mol)],
            ["Molecular Formula", properties["Molecular Formula"]],
            ["Molecular Weight", str(properties["Molecular Weight"]) + " g/mol"]
        ]
        basic_table = Table(basic_data, colWidths=[1.8 * inch, 4.9 * inch])
        basic_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF0F8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 5)
        ]))
        story += [basic_table, Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Molecular Structures", heading_style))
        img2d = save_2d_structure_png(mol)
        img3d = save_3d_structure_png(mol)
        if img2d:
            temp_images.append(img2d)
        if img3d:
            temp_images.append(img3d)

        if img2d and img3d:
            structure_table = Table([[
                Image(img2d, width=3.05 * inch, height=2.0 * inch),
                Image(img3d, width=3.05 * inch, height=2.0 * inch)
            ]], colWidths=[3.25 * inch, 3.25 * inch])
            structure_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER")
            ]))
            story += [structure_table, Spacer(1, 0.12 * inch)]
        elif img2d:
            story += [Image(img2d, width=4.8 * inch, height=3.2 * inch),
                      Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Molecular Properties and Interpretation", heading_style))
        prop_data = [[
            Paragraph("<b>Property</b>", header_style),
            Paragraph("<b>Value</b>", header_style),
            Paragraph("<b>Interpretation</b>", header_style)
        ]]
        for key, value in properties.items():
            prop_data.append([
                Paragraph(str(key), small_style),
                Paragraph(str(value), small_style),
                Paragraph(get_property_interpretation(key, value, properties), small_style)
            ])
        prop_table = Table(
            prop_data, colWidths=[1.55 * inch, 0.9 * inch, 4.0 * inch], repeatRows=1
        )
        prop_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 3)
        ]))
        story += [prop_table, Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Key Interpretation", heading_style))
        for point in build_key_interpretation_points(properties):
            story.append(Paragraph(point, body_style))
            story.append(Spacer(1, 0.04 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Python Libraries Used and Their Importance", heading_style))
    library_data = [
        [Paragraph("<b>Library</b>", header_style),
         Paragraph("<b>Importance in this project</b>", header_style)],
        [Paragraph("Streamlit", small_style),
         Paragraph("Creates the interactive web app, input controls and PDF download interface.", small_style)],
        [Paragraph("RDKit", small_style),
         Paragraph("Parses SMILES, validates structures, creates molecular drawings and 3D conformers, and calculates molecular descriptors.", small_style)],
        [Paragraph("Pandas", small_style),
         Paragraph("Organizes molecular properties in tables and supports CSV export.", small_style)],
        [Paragraph("Matplotlib and NumPy", small_style),
         Paragraph("Process molecular coordinates and generate a clear 2D projection of 3D geometry for the report.", small_style)],
        [Paragraph("py3Dmol", small_style),
         Paragraph("Displays interactive 3D molecular structures in the web application.", small_style)],
        [Paragraph("ReportLab", small_style),
         Paragraph("Builds the downloadable PDF with tables, figures and explanatory text.", small_style)],
        [Paragraph("io, os, tempfile", small_style),
         Paragraph("Handle in-memory PDF data and temporary image files.", small_style)]
    ]
    lib_table = Table(library_data, colWidths=[1.8 * inch, 4.9 * inch], repeatRows=1)
    lib_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story += [lib_table, Spacer(1, 0.15 * inch)]

    story.append(Paragraph("Illustrative Python Code", heading_style))
    code_style = ParagraphStyle(
        "MultiReportCode", parent=styles["Code"], fontName="Courier",
        fontSize=7, leading=8.5, leftIndent=4, rightIndent=4
    )
    code_example = "\n".join([
        "from rdkit import Chem",
        "from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors",
        "",
        'smiles_list = ["CCO", "c1ccccc1"]  # Add more SMILES strings here',
        "",
        "for smiles in smiles_list:",
        "    mol = Chem.MolFromSmiles(smiles)",
        "    if mol is None:",
        '        print("Invalid SMILES:", smiles)',
        "        continue",
        "",
        '    print("SMILES:", smiles)',
        '    print("Formula:", rdMolDescriptors.CalcMolFormula(mol))',
        '    print("Molecular weight:", Descriptors.MolWt(mol))',
        '    print("LogP:", Crippen.MolLogP(mol))'
    ])
    story.append(Preformatted(code_example, code_style, maxLineLength=92))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "This compact code example demonstrates processing multiple SMILES in one run. "
        "The complete application includes additional code for visualization, descriptor "
        "interpretation, student information and PDF generation. Calculated descriptors "
        "are computational results and should be interpreted appropriately.",
        body_style
    ))

    try:
        doc.build(story)
        return buffer.getvalue()
    finally:
        buffer.close()
        for image_path in temp_images:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception:
                pass


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧪 Virtual Lab")
st.sidebar.caption("CHEMISTRY WORKSPACE  ·  v2.1")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📖 Theory",

        "🧬 Molecular Visualization",

        "📊 Molecular Properties Report",

        "📈 Structure–Property Analysis",

        "🧪 Molecule Library",

        "🧩 Functional Group Explorer",

        "🔬 Atom & Bond Explorer",

        "📊 Descriptor Dashboard",

        "🧪 Virtual Experiments",

        "🤖 Chemistry Tutor",

        "🎯 Progress & Quiz",

        "📝 Assessment",

        "📄 Final Project Report"

    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input("Registration Number", key="registration_number", help="Used only to associate your progress. The raw number is not stored in the local progress database.")

if "chem_lab_session_id" not in st.session_state:
    st.session_state.chem_lab_session_id = uuid.uuid4().hex
student_key = make_student_key(student_name, registration_number, st.session_state.chem_lab_session_id)
student_label = student_name.strip() or "Student"

# Session-level cache for immediate feedback; the storage module also persists
# activity and quiz summaries in SQLite when the local filesystem is durable.
if "chem_lab_activity" not in st.session_state:
    st.session_state.chem_lab_activity = []
if "chem_lab_quiz_scores" not in st.session_state:
    st.session_state.chem_lab_quiz_scores = []

def record_lab_activity(activity, details=""):
    """Record an explicit learning action in session state and local SQLite storage."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    label = st.session_state.get("student_name", "").strip() or "Student"
    item = {
        "timestamp": timestamp,
        "student": label,
        "activity": activity,
        "details": details,
    }
    st.session_state.chem_lab_activity.append(item)
    try:
        save_activity(student_key, label, timestamp, activity, details)
        st.session_state.chem_lab_storage_error = ""
    except Exception as exc:
        st.session_state.chem_lab_storage_error = str(exc)

st.sidebar.markdown("""
### 🎓 Learning Objectives

Students can learn:

- SMILES notation
- Molecular representation
- 2D visualization
- 3D visualization
- Molecular descriptors
- Structure-property relationships
""")


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    # ========================================================
    # INSTITUTIONAL HEADER / COLLEGE LOGO
    # ========================================================
    logo_path = os.path.join(os.path.dirname(__file__), "srm_logo.jpeg")

    if os.path.exists(logo_path):
        logo_col1, logo_col2, logo_col3 = st.columns([1, 4, 1])
        with logo_col2:
            st.image(logo_path, use_container_width=True)

    st.markdown(
        '<div class="main-title">🧪 Cheminformatics Virtual Laboratory</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; font-size:24px; font-weight:700; color:#1f4e79;">
        SRM Institute of Science and Technology, Tiruchirappalli
        </div>
        <div style="text-align:center; font-size:18px; font-weight:600; margin-top:5px;">
        Faculty of Engineering and Technology | School of Sciences | Division of Chemistry
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("## Welcome to the Virtual Laboratory")

    st.write("""
    This interactive virtual laboratory helps students understand
    important concepts in cheminformatics and computational chemistry.
    """)

    st.markdown("### 🧪 What can students do?")

    st.markdown("""
    ✅ Enter any SMILES notation

    ✅ Validate molecular structures

    ✅ Generate 2D structures

    ✅ Generate interactive 3D structures

    ✅ Select different molecular visualization styles

    ✅ Calculate complete molecular properties

    ✅ Download molecular reports

    ✅ Compare multiple molecules

    ✅ Complete an assessment
    """)

    st.markdown("### Example Molecules")

    example_df = pd.DataFrame({

        "Molecule": [

            "Ethanol",

            "Benzene",

            "Phenol",

            "Acetic Acid",

            "Aspirin",

            "Caffeine"
        ],

        "SMILES": [

            "CCO",

            "c1ccccc1",

            "Oc1ccccc1",

            "CC(=O)O",

            "CC(=O)Oc1ccccc1C(=O)O",

            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        ]
    })

    st.dataframe(
        example_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Developed by")

    st.markdown("""
    **GURUPRASHATH LV**  
    Register Number: RA2611030050012  
    1st Year B.Tech CSE – Cyber Security Student  
    SRM Institute of Science and Technology (SRMIST), Tiruchirappalli Campus
    """)

    st.divider()

    st.markdown("""
    **SIVASANKARI**  
    Register Number: RA2611030050057  
    1st Year B.Tech CSE – Cyber Security Student  
    SRM Institute of Science and Technology (SRMIST), Tiruchirappalli Campus
    """)


# ============================================================
# THEORY PAGE
# ============================================================

elif page == "📖 Theory":

    st.title("📖 Introduction to Cheminformatics")

    st.markdown("""
## What is Cheminformatics?

Cheminformatics combines:

- Chemistry ⚗️
- Computer Science 💻
- Data Analysis 📊

It uses computational techniques to store, process,
visualize and analyze chemical information.
""")

    st.markdown("""
## SMILES

**SMILES** means:

### Simplified Molecular Input Line Entry System

It represents a molecular structure using a text string.

Examples:

| Molecule | SMILES |
|---|---|
| Ethanol | CCO |
| Benzene | c1ccccc1 |
| Acetic Acid | CC(=O)O |
""")

    st.markdown("""
## Molecular Descriptors

Molecular descriptors provide numerical information
about a chemical structure.

Examples include:

- Molecular Weight
- LogP
- TPSA
- Hydrogen Bond Donors
- Hydrogen Bond Acceptors
- Rotatable Bonds
""")

    st.markdown("""
## 3D Molecular Models

Different visualization models help students understand
the molecular geometry.

### 🟢 Ball and Stick

Shows atoms as spheres and bonds as sticks.

### 🟡 Stick Model

Emphasizes chemical bonding.

### 🔵 Space Filling

Shows the approximate molecular volume.

### ⚫ Wireframe

Provides a simplified representation of the molecule.
""")


# ============================================================
# MOLECULAR VISUALIZATION PAGE
# ============================================================

elif page == "🧬 Molecular Visualization":

    st.title("🧬 Molecular Visualization Laboratory")

    st.info("Enter a new SMILES or load an example. The selected molecule will be used across the laboratory pages.")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles = st.text_input(
            "Enter SMILES",
            key="active_smiles"
        )

    with col2:
        st.selectbox(
            "Choose Example Molecule",
            list(EXAMPLE_MOLECULES.keys()),
            key="selected_example"
        )

        st.button(
            "Load Example",
            on_click=load_shared_example,
            key="load_visualization_example"
        )

    # Keep the latest valid molecule available for the other pages.
    mol = get_molecule(smiles)

    if mol is None:

        st.error("❌ Invalid SMILES notation.")

    else:

        st.success("✅ Valid Molecular Structure!")
        st.session_state.latest_smiles = smiles
        st.session_state.latest_properties = calculate_properties(mol)

        col1, col2 = st.columns([1.2, 1])

        with col1:

            st.subheader("2D Molecular Structure")

            image = generate_2d_image(mol)

            st.image(image)

        with col2:

            st.subheader("Molecular Information")

            st.write(
                "**Input SMILES:**",
                smiles
            )

            st.write(
                "**Canonical SMILES:**",
                Chem.MolToSmiles(mol)
            )

            st.write(
                "**Formula:**",
                rdMolDescriptors.CalcMolFormula(mol)
            )

            bond_info = calculate_bond_information(mol)

            st.write(
                "**Heavy Atoms:**",
                mol.GetNumHeavyAtoms()
            )

            st.write(
                "**Total Atoms (including H):**",
                bond_info["Total Atoms Including H"]
            )

            st.write(
                "**Heavy-Atom Bonds:**",
                bond_info["Heavy Atom Bonds"]
            )

            st.write(
                "**Sigma (σ) Bonds:**",
                bond_info["Sigma Bonds"]
            )

            st.write(
                "**Pi (π) Bonds:**",
                bond_info["Pi Bonds"]
            )

            st.write(
                "**Total Bonds (σ + π):**",
                bond_info["Total Bonds"]
            )

        st.divider()

        st.subheader("🌐 Interactive 3D Molecular Structure")

        visualization_style = st.selectbox(

            "Select Molecular Model",

            [

                "Ball and Stick",

                "Stick",

                "Space Filling",

                "Wireframe",

                "Cartoon"
            ]
        )

        show_surface = st.checkbox(
            "Show van der Waals Surface"
        )

        if st.button("🚀 Generate 3D Structure"):

            with st.spinner(
                "Generating 3D molecular structure..."
            ):

                mol3d = generate_3d_molecule(mol)

            if mol3d is None:

                st.error(
                    "Unable to generate the 3D structure."
                )

            else:

                display_3d_structure(

                    mol3d,

                    visualization_style,

                    show_surface
                )


# ============================================================
# COMPLETE MOLECULAR PROPERTIES REPORT
# ============================================================

elif page == "📊 Molecular Properties Report":

    st.title("📊 Complete Molecular Properties Report")

    st.info("""
Enter any valid SMILES notation. The system will automatically
generate a complete molecular properties report.
""")

    st.write("### Select or Enter Your Molecule")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles = st.text_input(
            "Enter New SMILES",
            key="active_smiles"
        )

    with col2:
        st.selectbox(
            "Choose Example Molecule",
            list(EXAMPLE_MOLECULES.keys()),
            key="selected_example"
        )

        st.button(
            "Load Example",
            on_click=load_shared_example,
            key="load_properties_example"
        )

    st.caption("Once you enter a valid SMILES or load an example, the complete 2D/3D structures and molecular properties below are generated for the same molecule.")

    if smiles:

        mol = get_molecule(smiles)

        if mol is None:

            st.error(
                "❌ Invalid SMILES notation. Please check the structure."
            )

        else:

            st.success(
                "✅ Valid Molecular Structure Detected!"
            )

            properties = calculate_properties(mol)

            canonical_smiles = Chem.MolToSmiles(mol)

            # Save the latest student-entered molecule for the Final Project Report
            st.session_state.latest_smiles = smiles
            st.session_state.latest_properties = properties

            # ------------------------------------------------
            # BASIC INFORMATION
            # ------------------------------------------------

            col1, col2 = st.columns([1.2, 1])

            with col1:

                st.subheader("🧬 2D Molecular Structure")

                image = generate_2d_image(mol)

                st.image(image)

            with col2:

                st.subheader("📋 Basic Information")

                st.write(
                    "**Input SMILES:**",
                    smiles
                )

                st.write(
                    "**Canonical SMILES:**",
                    canonical_smiles
                )

                st.write(
                    "**Molecular Formula:**",
                    properties["Molecular Formula"]
                )

                st.write(
                    "**Atoms:**",
                    properties["Number of Atoms"]
                )

                st.write(
                    "**Heavy Atoms:**",
                    properties["Heavy Atoms"]
                )

                st.write(
                    "**Heavy-Atom Bonds:**",
                    properties["Heavy Atom Bonds"]
                )

                st.write(
                    "**Sigma (σ) Bonds:**",
                    properties["Sigma (σ) Bonds"]
                )

                st.write(
                    "**Pi (π) Bonds:**",
                    properties["Pi (π) Bonds"]
                )

                st.write(
                    "**Total Bonds (σ + π):**",
                    properties["Total Bonds (σ + π)"]
                )

            st.divider()

            # ------------------------------------------------
            # PROPERTY METRICS
            # ------------------------------------------------

            st.subheader("📊 Important Molecular Properties")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Molecular Weight",
                f"{properties['Molecular Weight']} g/mol"
            )

            col2.metric(
                "LogP",
                properties["LogP"]
            )

            col3.metric(
                "TPSA",
                f"{properties['TPSA']} Å²"
            )

            col4.metric(
                "Ring Count",
                properties["Ring Count"]
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "H-Bond Donors",
                properties["Hydrogen Bond Donors"]
            )

            col2.metric(
                "H-Bond Acceptors",
                properties["Hydrogen Bond Acceptors"]
            )

            col3.metric(
                "Rotatable Bonds",
                properties["Rotatable Bonds"]
            )

            col4.metric(
                "Fraction Csp3",
                properties["Fraction Csp3"]
            )

            st.divider()

            # ------------------------------------------------
            # COMPLETE TABLE
            # ------------------------------------------------

            st.subheader("📑 Complete Molecular Property Table")

            report_data = [

                ["Input SMILES", smiles],

                ["Canonical SMILES", canonical_smiles]
            ]

            for key, value in properties.items():

                report_data.append(
                    [key, value]
                )

            report_df = pd.DataFrame(

                report_data,

                columns=[
                    "Property",
                    "Value"
                ]
            )

            st.dataframe(

                report_df,

                use_container_width=True,

                hide_index=True
            )

            # ------------------------------------------------
            # DOWNLOAD CSV
            # ------------------------------------------------

            csv = report_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                "⬇️ Download Properties Report (CSV)",

                data=csv,

                file_name="Molecular_Properties_Report.csv",

                mime="text/csv"
            )

            # ------------------------------------------------
            # DOWNLOAD PDF
            # ------------------------------------------------

            if st.button(
                "📄 Generate Molecular PDF Report"
            ):

                with st.spinner(
                    "Generating molecular PDF report..."
                ):

                    pdf = generate_molecule_pdf(

                        smiles,

                        mol,

                        properties
                    )

                st.download_button(

                    "⬇️ Download Molecular Properties PDF",

                    data=pdf,

                    file_name="Molecular_Properties_Report.pdf",

                    mime="application/pdf"
                )

            st.divider()

            # ------------------------------------------------
            # 3D VISUALIZATION
            # ------------------------------------------------

            st.subheader("🌐 3D Molecular Visualization")

            model_style = st.selectbox(

                "Select 3D Model",

                [

                    "Ball and Stick",

                    "Stick",

                    "Space Filling",

                    "Wireframe",

                    "Cartoon"
                ],

                key="properties_3d_style"
            )

            surface = st.checkbox(

                "Show van der Waals Surface",

                key="properties_surface"
            )

            if st.button(
                "🚀 Generate 3D Molecular Model",
                key="properties_generate_3d"
            ):

                with st.spinner(
                    "Generating 3D molecular model..."
                ):

                    mol3d = generate_3d_molecule(mol)

                if mol3d is None:

                    st.error(
                        "Unable to generate the 3D structure."
                    )

                else:

                    display_3d_structure(

                        mol3d,

                        model_style,

                        surface
                    )


# ============================================================
# STRUCTURE PROPERTY ANALYSIS
# ============================================================

elif page == "📈 Structure–Property Analysis":

    st.title("📈 Structure–Property Analysis")

    st.write("""
Compare the properties of multiple molecules.
Enter one molecule per line in the format:

**Molecule Name, SMILES**
""")

    default_data = """Ethanol,CCO
Benzene,c1ccccc1
Phenol,Oc1ccccc1
Acetic Acid,CC(=O)O
Aspirin,CC(=O)Oc1ccccc1C(=O)O"""

    molecule_input = st.text_area(

        "Enter Molecules",

        value=default_data,

        height=200
    )

    rows = []

    for line in molecule_input.splitlines():

        if "," not in line:

            continue

        name, smiles = line.split(",", 1)

        mol = get_molecule(smiles)

        if mol:

            prop = calculate_properties(mol)

            rows.append({

                "Molecule": name.strip(),

                "MW": prop["Molecular Weight"],

                "LogP": prop["LogP"],

                "TPSA": prop["TPSA"],

                "HBD": prop["Hydrogen Bond Donors"],

                "HBA": prop["Hydrogen Bond Acceptors"],

                "Rotatable Bonds": prop["Rotatable Bonds"],

                "Ring Count": prop["Ring Count"]
            })

    if rows:

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("📋 Comparison Table")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Comparison Table (CSV)",
            data=csv_data,
            file_name="structure_property_comparison.csv",
            mime="text/csv",
            key="download_structure_property_csv"
        )

    else:
        st.warning("Please enter at least one valid molecule in the format: Molecule Name, SMILES")


# ============================================================
# ASSESSMENT
# ============================================================

elif page == "📝 Assessment":

    st.title("📝 Cheminformatics Assessment")

    questions = [
        {"question":"What is the full form of SMILES?", "options":["Simplified Molecular Input Line Entry System","Standard Molecular Information Language Encoding System","Simple Molecular Identification and Labeling System"], "answer":"Simplified Molecular Input Line Entry System"},
        {"question":"What does a SMILES string represent?", "options":["A molecular structure using text","A spectroscopy technique","A laboratory instrument"], "answer":"A molecular structure using text"},
        {"question":"Which SMILES represents ethanol?", "options":["CCO","c1ccccc1","CC(=O)O"], "answer":"CCO"},
        {"question":"Which SMILES represents benzene?", "options":["c1ccccc1","CCO","O=C=O"], "answer":"c1ccccc1"},
        {"question":"In SMILES, parentheses are mainly used to indicate:", "options":["Branches","Molecular weight","Atom colour"], "answer":"Branches"},
        {"question":"Which property is commonly associated with molecular lipophilicity?", "options":["LogP","TPSA","HBD"], "answer":"LogP"},
        {"question":"What does TPSA describe?", "options":["Topological Polar Surface Area","Total Pi Surface Area","Thermal Property Surface Analysis"], "answer":"Topological Polar Surface Area"},
        {"question":"What does HBA mean?", "options":["Hydrogen Bond Acceptor","Heavy Bond Atom","Hydrogen Bond Analysis"], "answer":"Hydrogen Bond Acceptor"},
        {"question":"What does HBD mean?", "options":["Hydrogen Bond Donor","Heavy Bond Descriptor","Hydrogen Bond Distance"], "answer":"Hydrogen Bond Donor"},
        {"question":"A double bond contains how many pi (π) bonds?", "options":["1","2","0"], "answer":"1"},
        {"question":"A triple bond contains how many pi (π) bonds?", "options":["2","1","3"], "answer":"2"},
        {"question":"Which model is useful for visualizing approximate molecular volume?", "options":["Space Filling","Wireframe only","Text SMILES"], "answer":"Space Filling"}
    ]

    student_answers = []

    for i, q in enumerate(questions):

        answer = st.radio(
            q["question"],
            q["options"],
            index=None,
            key=f"assessment_v2_{i}"
        )

        student_answers.append(answer)

    if st.button("Submit Assessment"):

        if any(answer is None for answer in student_answers):
            st.warning("Please answer all questions before submitting the assessment.")
            st.stop()

        score = 0

        for i, q in enumerate(questions):

            if student_answers[i] == q["answer"]:
                score += 1

        percentage = (score / len(questions)) * 100

        st.success(
            f"Your Score: {score}/{len(questions)}"
        )

        st.metric(
            "Percentage",
            f"{percentage:.1f}%"
        )

        if percentage >= 80:

            st.balloons()

            st.success("🎉 Excellent performance!")

        elif percentage >= 50:

            st.info("👍 Good! Continue practicing.")

        else:

            st.warning(
                "📚 Please review the Theory section and try again."
            )

# ============================================================
# FINAL PROJECT REPORT
# ============================================================

elif page == "📄 Final Project Report":

    st.title("📄 Final Student Molecular Analysis Report")
    st.write(
        "Enter one SMILES string per line. The report will include every valid molecule "
        "in a single downloadable PDF."
    )

    name = st.text_input(
        "Student Name for Report",
        value=st.session_state.get("student_name", ""),
        key="final_report_name"
    )
    reg = st.text_input(
        "Registration Number for Report",
        value=st.session_state.get("registration_number", ""),
        key="final_report_reg"
    )

    default_smiles = st.session_state.get(
        "latest_smiles",
        st.session_state.get("active_smiles", "CCO")
    )
    multi_smiles_text = st.text_area(
        "Enter SMILES (one molecule per line)",
        value=st.session_state.get("final_report_smiles_multi", default_smiles),
        height=150,
        key="final_report_smiles_multi",
        help="Example: CCO on the first line, c1ccccc1 on the second line, and so on."
    )

    smiles_entries = [
        line.strip() for line in multi_smiles_text.splitlines() if line.strip()
    ]
    molecule_records = []
    invalid_smiles = []

    for smiles_item in smiles_entries:
        mol_item = get_molecule(smiles_item)
        if mol_item is None:
            invalid_smiles.append(smiles_item)
            continue
        props_item = calculate_properties(mol_item)
        molecule_records.append((smiles_item, mol_item, props_item))

    if invalid_smiles:
        st.warning(
            "These SMILES could not be interpreted and will not be included: "
            + ", ".join(invalid_smiles)
        )

    if molecule_records:
        st.success(f"{len(molecule_records)} valid molecule(s) ready for the report.")
        preview_rows = []
        for idx, (smiles_item, mol_item, props_item) in enumerate(molecule_records, start=1):
            preview_rows.append({
                "No.": idx,
                "Input SMILES": smiles_item,
                "Molecular Formula": props_item["Molecular Formula"],
                "Molecular Weight (g/mol)": props_item["Molecular Weight"]
            })
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        if st.button("📄 Generate Single PDF for All Molecules"):
            with st.spinner(
                f"Generating one PDF report for {len(molecule_records)} molecule(s)..."
            ):
                pdf = generate_multi_final_report(name, reg, molecule_records)
            st.success("✅ Combined molecular report generated successfully!")
            st.download_button(
                label="⬇️ Download Combined Molecular Report PDF",
                data=pdf,
                file_name="Cheminformatics_Multi_Molecule_Report.pdf",
                mime="application/pdf",
                key="download_multi_molecule_pdf"
            )
    else:
        st.info("Enter at least one valid SMILES string to generate the report.")


# ============================================================
# UPGRADE MODULE 1: MOLECULE LIBRARY
# ============================================================

elif page == "🧪 Molecule Library":

    st.title("🧪 Molecule Library")
    st.write("Explore a curated starter library of common molecules. Select one to load its SMILES into the shared molecule field.")

    molecule_library = {
        "Water": ("O", "Small inorganic molecule; a common solvent."),
        "Methane": ("C", "Simplest alkane."),
        "Ethanol": ("CCO", "An alcohol containing two carbon atoms."),
        "Acetone": ("CC(=O)C", "A ketone commonly used as a solvent."),
        "Acetic Acid": ("CC(=O)O", "A simple carboxylic acid."),
        "Formaldehyde": ("C=O", "The simplest aldehyde."),
        "Ethene": ("C=C", "A hydrocarbon containing a carbon–carbon double bond."),
        "Ethyne": ("C#C", "A hydrocarbon containing a carbon–carbon triple bond."),
        "Benzene": ("c1ccccc1", "An aromatic six-membered carbon ring."),
        "Phenol": ("Oc1ccccc1", "An aromatic ring bearing a hydroxyl group."),
        "Aniline": ("Nc1ccccc1", "An aromatic amine."),
        "Toluene": ("Cc1ccccc1", "Methyl-substituted benzene."),
        "Glycine": ("NCC(=O)O", "The simplest proteinogenic amino acid."),
        "Lactic Acid": ("CC(O)C(=O)O", "An alpha-hydroxy carboxylic acid."),
        "Urea": ("NC(=O)N", "A compound containing a carbonyl bonded to two nitrogens."),
        "Aspirin": ("CC(=O)Oc1ccccc1C(=O)O", "Acetylsalicylic acid."),
        "Caffeine": ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", "A naturally occurring methylxanthine."),
        "Paracetamol": ("CC(=O)Nc1ccc(O)cc1", "Also known as acetaminophen."),
        "Ibuprofen": ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "A common non-steroidal anti-inflammatory drug."),
        "Nicotinamide": ("NC(=O)c1ccncc1", "A pyridine carboxamide related to vitamin B3.")
    }

    selected_library_molecule = st.selectbox("Choose a molecule", list(molecule_library.keys()))
    lib_smiles, lib_description = molecule_library[selected_library_molecule]
    lib_mol = get_molecule(lib_smiles)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(selected_library_molecule)
        st.write(lib_description)
        st.code(lib_smiles, language="text")
        st.write("**Formula:**", rdMolDescriptors.CalcMolFormula(lib_mol))
        st.write("**Molecular weight:**", f"{Descriptors.MolWt(lib_mol):.3f} g/mol")
        if st.button("Load into shared molecule input", key="load_from_library"):
            st.session_state.active_smiles = lib_smiles
            st.session_state.selected_example = "Ethanol" if lib_smiles == "CCO" else st.session_state.get("selected_example", "Ethanol")
            st.session_state.latest_smiles = lib_smiles
            st.success("SMILES loaded. Open Molecular Visualization or Molecular Properties Report.")
    with col2:
        st.image(Draw.MolToImage(lib_mol, size=(480, 360)), caption=f"2D structure of {selected_library_molecule}", use_container_width=True)

    st.divider()
    st.subheader("Browse all library entries")
    lib_rows = []
    for mol_name, (mol_smiles, description) in molecule_library.items():
        mol_obj = get_molecule(mol_smiles)
        lib_rows.append({
            "Molecule": mol_name,
            "Formula": rdMolDescriptors.CalcMolFormula(mol_obj),
            "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol_obj), 3),
            "SMILES": mol_smiles,
            "Description": description
        })
    st.dataframe(pd.DataFrame(lib_rows), use_container_width=True, hide_index=True)
    st.download_button("Download molecule library (CSV)", pd.DataFrame(lib_rows).to_csv(index=False).encode("utf-8"),
                       "chemistry_molecule_library.csv", "text/csv")


# ============================================================
# UPGRADE MODULE 2: FUNCTIONAL GROUP EXPLORER
# ============================================================

elif page == "🧩 Functional Group Explorer":

    st.title("🧩 Functional Group Explorer")
    st.write("Identify common functional groups in a molecule using RDKit substructure patterns. Matches are educational indicators, not a complete chemical interpretation.")

    functional_groups = {
        "Alcohol (C–OH)": "[CX4][OX2H]",
        "Phenol (Ar–OH)": "c[OX2H]",
        "Carboxylic acid (–COOH)": "[CX3](=O)[OX2H1]",
        "Ester (–COO–)": "[CX3](=O)[OX2][#6]",
        "Aldehyde (–CHO)": "[CX3H1](=O)[#6]",
        "Ketone (>C=O)": "[#6][CX3](=O)[#6]",
        "Amide (–CON–)": "[CX3](=O)[NX3]",
        "Amine (nitrogen group)": "[NX3;!$(NC=O)]",
        "Nitrile (–C≡N)": "[C]#N",
        "Alkene (C=C)": "[CX3]=[CX3]",
        "Alkyne (C≡C)": "[CX2]#[CX2]",
        "Aromatic ring": "a1aaaaa1",
        "Halogen attached to carbon": "[#6][F,Cl,Br,I]",
        "Nitro group (–NO₂)": "[N+](=O)[O-]",
        "Ether (C–O–C)": "[OD2]([#6])[#6]"
    }

    fg_smiles = st.text_input("Enter SMILES to analyze", value=st.session_state.get("active_smiles", "CCO"), key="functional_group_smiles")
    fg_mol = get_molecule(fg_smiles)
    if not fg_smiles.strip():
        st.info("Enter a SMILES string to begin.")
    elif fg_mol is None:
        st.error("Invalid SMILES. Check the notation and try again.")
    else:
        found = []
        highlight_atoms = set()
        for fg_name, smarts in functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            matches = fg_mol.GetSubstructMatches(pattern)
            if matches:
                found.append({"Functional Group": fg_name, "Matches": len(matches), "SMARTS Pattern": smarts})
                for match in matches:
                    highlight_atoms.update(match)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(Draw.MolToImage(fg_mol, size=(600, 430), highlightAtoms=list(highlight_atoms)),
                     caption="Matched atoms are highlighted", use_container_width=True)
        with c2:
            st.metric("Functional group types found", len(found))
            st.metric("Total pattern matches", sum(row["Matches"] for row in found))
            if found:
                st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
            else:
                st.info("No groups from this starter pattern set were detected.")
        st.caption("SMARTS patterns can overlap, and some patterns are intentionally broad. Verify results against the molecular structure and chemistry references.")


# ============================================================
# UPGRADE MODULE 3: ATOM AND BOND EXPLORER
# ============================================================

elif page == "🔬 Atom & Bond Explorer":

    st.title("🔬 Atom & Bond Explorer")
    st.write("Inspect atom indices, elements, charges, aromaticity, and bond order for a molecular structure.")

    ab_smiles = st.text_input("Enter SMILES", value=st.session_state.get("active_smiles", "CCO"), key="atom_bond_smiles")
    ab_mol = get_molecule(ab_smiles)
    if not ab_smiles.strip():
        st.info("Enter a SMILES string to explore its atoms and bonds.")
    elif ab_mol is None:
        st.error("Invalid SMILES. Please check the notation.")
    else:
        atom_rows = []
        for atom in ab_mol.GetAtoms():
            atom_rows.append({
                "Atom Index": atom.GetIdx(),
                "Element": atom.GetSymbol(),
                "Atomic Number": atom.GetAtomicNum(),
                "Formal Charge": atom.GetFormalCharge(),
                "Degree": atom.GetDegree(),
                "Total H Count": atom.GetTotalNumHs(),
                "Aromatic": atom.GetIsAromatic(),
                "Hybridization": str(atom.GetHybridization()),
                "In Ring": atom.IsInRing()
            })
        bond_rows = []
        for bond in ab_mol.GetBonds():
            bond_rows.append({
                "Bond Index": bond.GetIdx(),
                "Atom 1 Index": bond.GetBeginAtomIdx(),
                "Atom 1": bond.GetBeginAtom().GetSymbol(),
                "Atom 2 Index": bond.GetEndAtomIdx(),
                "Atom 2": bond.GetEndAtom().GetSymbol(),
                "Bond Type": str(bond.GetBondType()),
                "Bond Order": bond.GetBondTypeAsDouble(),
                "Aromatic": bond.GetIsAromatic(),
                "In Ring": bond.IsInRing()
            })

        st.image(Draw.MolToImage(ab_mol, size=(700, 420), kekulize=True), caption="Molecular structure", use_container_width=True)
        st.subheader(f"Atoms ({len(atom_rows)})")
        atom_df = pd.DataFrame(atom_rows)
        st.dataframe(atom_df, use_container_width=True, hide_index=True)
        st.download_button("Download atom table (CSV)", atom_df.to_csv(index=False).encode("utf-8"),
                           "atom_details.csv", "text/csv")
        st.subheader(f"Bonds ({len(bond_rows)})")
        bond_df = pd.DataFrame(bond_rows)
        if bond_rows:
            st.dataframe(bond_df, use_container_width=True, hide_index=True)
            st.download_button("Download bond table (CSV)", bond_df.to_csv(index=False).encode("utf-8"),
                               "bond_details.csv", "text/csv")
        else:
            st.info("This molecule has no explicit atom-to-atom bonds in its SMILES representation.")
        st.caption("Atom indices start at 0 and refer to the parsed SMILES graph. Hydrogens may be implicit unless explicitly added.")


# ============================================================
# UPGRADE MODULE 4: DESCRIPTOR DASHBOARD
# ============================================================

elif page == "📊 Descriptor Dashboard":

    st.title("📊 Molecular Descriptor Dashboard")
    st.write("Compare common molecular descriptors across a set of SMILES strings and visualize the differences.")

    dashboard_default = """Ethanol,CCO
Acetone,CC(=O)C
Benzene,c1ccccc1
Phenol,Oc1ccccc1
Aspirin,CC(=O)Oc1ccccc1C(=O)O
Caffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O"""
    dashboard_text = st.text_area("Enter one molecule per line as Name,SMILES", value=dashboard_default, height=180, key="descriptor_dashboard_input")
    dashboard_rows, dashboard_errors = [], []
    for line_no, line in enumerate(dashboard_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if "," not in line:
            dashboard_errors.append(f"Line {line_no}: expected Name,SMILES")
            continue
        mol_name, mol_smiles = line.split(",", 1)
        mol_obj = get_molecule(mol_smiles)
        if mol_obj is None:
            dashboard_errors.append(f"Line {line_no}: invalid SMILES for {mol_name.strip()}")
            continue
        props = calculate_properties(mol_obj)
        dashboard_rows.append({
            "Molecule": mol_name.strip() or f"Molecule {line_no}",
            "Formula": props["Molecular Formula"],
            "Molecular Weight": props["Molecular Weight"],
            "LogP": props["LogP"],
            "TPSA": props["TPSA"],
            "H-Bond Donors": props["Hydrogen Bond Donors"],
            "H-Bond Acceptors": props["Hydrogen Bond Acceptors"],
            "Rotatable Bonds": props["Rotatable Bonds"],
            "Ring Count": props["Ring Count"],
            "Fraction Csp3": props["Fraction Csp3"]
        })
    for issue in dashboard_errors:
        st.warning(issue)
    if dashboard_rows:
        dash_df = pd.DataFrame(dashboard_rows)
        st.dataframe(dash_df, use_container_width=True, hide_index=True)
        st.download_button("Download descriptor dataset (CSV)", dash_df.to_csv(index=False).encode("utf-8"),
                           "molecular_descriptor_dashboard.csv", "text/csv")
        numeric_columns = ["Molecular Weight", "LogP", "TPSA", "H-Bond Donors", "H-Bond Acceptors", "Rotatable Bonds", "Ring Count", "Fraction Csp3"]
        selected_descriptors = st.multiselect("Choose descriptors to plot", numeric_columns, default=["Molecular Weight", "LogP", "TPSA"])
        if selected_descriptors:
            plot_df = dash_df.set_index("Molecule")[selected_descriptors]
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_df.plot(kind="bar", ax=ax)
            ax.set_title("Molecular descriptor comparison")
            ax.set_xlabel("Molecule")
            ax.set_ylabel("Descriptor value (units differ by descriptor)")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.caption("Note: plotted descriptors may use different units and scales. For strict comparisons, plot one descriptor at a time.")
    else:
        st.info("Enter at least one valid molecule as Name,SMILES.")


# ============================================================
# UPGRADE MODULE 5: VIRTUAL EXPERIMENTS
# ============================================================

elif page == "🧪 Virtual Experiments":
    st.title("🧪 Virtual Chemistry Experiments")
    st.write("Run guided, browser-based numerical simulations. These are educational models, not live laboratory equipment.")
    exp_tab1, exp_tab2, exp_tab3, exp_tab4, exp_tab5 = st.tabs(["Acid–base titration", "Reaction kinetics", "Electrochemistry", "Beer–Lambert law", "Arrhenius equation"])

    with exp_tab1:
        st.subheader("Strong acid–strong base titration")
        st.caption("Model: monoprotic strong acid titrated with a strong base at approximately 25 °C; ideal-solution assumptions apply.")
        c1, c2, c3 = st.columns(3)
        with c1:
            acid_conc = st.number_input("Acid concentration (mol/L)", min_value=0.001, max_value=5.0, value=0.1, step=0.01, key="titr_ac")
            acid_vol = st.number_input("Initial acid volume (mL)", min_value=1.0, max_value=500.0, value=25.0, step=1.0, key="titr_av")
        with c2:
            base_conc = st.number_input("Base concentration (mol/L)", min_value=0.001, max_value=5.0, value=0.1, step=0.01, key="titr_bc")
            max_factor = st.slider("Plot range (× equivalence volume)", min_value=1.2, max_value=2.5, value=2.0, step=0.1, key="titr_range")
        with c3:
            titration_temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0, step=1.0, key="titr_temp")
            st.metric("Acid amount", f"{acid_conc * acid_vol:.4f} mmol")
            eq_vol = acid_conc * acid_vol / base_conc
            st.metric("Equivalence volume", f"{eq_vol:.3f} mL")
        import numpy as np
        volumes = np.linspace(0, eq_vol * max_factor, 251)
        acid_mol = acid_conc * acid_vol / 1000.0
        ph_values = []
        for v_base in volumes:
            ph_values.append(strong_acid_strong_base_ph(acid_conc, acid_vol, base_conc, float(v_base), titration_temp))
        titr_df = pd.DataFrame({"Base added (mL)": volumes, "pH (ideal model)": ph_values})
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(volumes, ph_values, label="Calculated pH", linewidth=2.5)
        ax.axvline(eq_vol, linestyle="--", label=f"Equivalence: {eq_vol:.2f} mL")
        ax.set_xlabel("Base added (mL)"); ax.set_ylabel("pH")
        ax.set_ylim(min(0.0, min(ph_values) - 0.5), max(14.0, max(ph_values) + 0.5))
        ax.set_title("Strong acid–strong base titration curve", pad=12); ax.grid(True, alpha=0.22); ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        chosen_v = st.slider("Inspect a base volume (mL)", min_value=0.0, max_value=float(max(0.1, eq_vol * max_factor)), value=float(eq_vol), step=0.01, key="titr_inspect")
        selected_ph = strong_acid_strong_base_ph(acid_conc, acid_vol, base_conc, chosen_v, titration_temp)
        st.metric("Calculated pH at selected volume", f"{selected_ph:.4f}")
        st.caption("Calculation uses excess strong acid/base plus water autoionization. Values are ideal estimates; concentrated solutions may require activity corrections.")
        st.download_button("Download titration data (CSV)", titr_df.to_csv(index=False).encode("utf-8"), "titration_curve.csv", "text/csv")
        if st.button("Record titration experiment", key="record_titration"):
            record_lab_activity("Virtual experiment", f"Strong acid–strong base titration; equivalence volume {eq_vol:.2f} mL")
            st.success("Experiment added to your session activity log.")

    with exp_tab2:
        st.subheader("First-order reaction kinetics")
        st.caption("Integrated rate law: [A]ₜ = [A]₀ e⁻ᵏᵗ. This model assumes a single first-order process and constant temperature.")
        k1, k2, k3 = st.columns(3)
        with k1:
            c0 = st.number_input("Initial concentration [A]₀ (mol/L)", min_value=0.01, max_value=10.0, value=1.0, step=0.1, key="kin_c0")
        with k2:
            rate_k = st.number_input("Rate constant k (s⁻¹)", min_value=0.0001, max_value=2.0, value=0.05, step=0.01, format="%.4f", key="kin_k")
        with k3:
            end_t = st.number_input("Simulation duration (s)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0, key="kin_t")
        times = np.linspace(0, end_t, 201)
        concentrations = np.array([first_order_concentration(c0, rate_k, float(t)) for t in times])
        kin_df = pd.DataFrame({"Time (s)": times, "Concentration (mol/L)": concentrations, "ln([A]/[A]0)": -rate_k * times})
        fig, ax = plt.subplots(figsize=(9, 4.2)); ax.plot(times, concentrations, label="[A] remaining")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Concentration (mol/L)"); ax.set_title("First-order concentration vs time"); ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        half_life = first_order_half_life(rate_k)
        st.metric("Half-life", f"{half_life:.3g} s")
        st.metric("Concentration at final time", f"{concentrations[-1]:.5g} mol/L")
        fig, ax = plt.subplots(figsize=(9, 3.5)); ax.plot(times, -rate_k * times); ax.set_xlabel("Time (s)"); ax.set_ylabel("ln([A]/[A]₀)"); ax.set_title("First-order linearized plot"); ax.grid(True, alpha=0.25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.download_button("Download kinetics data (CSV)", kin_df.to_csv(index=False).encode("utf-8"), "first_order_kinetics.csv", "text/csv")
        if st.button("Record kinetics experiment", key="record_kinetics"):
            record_lab_activity("Virtual experiment", f"First-order kinetics; k={rate_k:g} s^-1; half-life={half_life:.3g} s")
            st.success("Experiment added to your session activity log.")

    with exp_tab3:
        st.subheader("Nernst equation calculator")
        st.caption("For a specified reaction quotient Q: E = E° − (RT/nF) ln(Q). The calculator does not infer Q from a full cell reaction.")
        e1, e2, e3, e4 = st.columns(4)
        with e1: e0 = st.number_input("Standard potential E° (V)", min_value=-5.0, max_value=5.0, value=1.10, step=0.01, key="nernst_e0")
        with e2: electrons = st.number_input("Electrons transferred n", min_value=1, max_value=12, value=2, step=1, key="nernst_n")
        with e3: quotient = st.number_input("Reaction quotient Q", min_value=0.000001, max_value=1000000.0, value=1.0, format="%.6g", key="nernst_q")
        with e4: temp_c = st.number_input("Temperature (°C)", min_value=-50.0, max_value=200.0, value=25.0, step=1.0, key="nernst_temp")
        R, F = 8.314462618, 96485.33212
        potential = nernst_potential(e0, int(electrons), quotient, temp_c)
        st.metric("Calculated cell potential E", f"{potential:.4f} V")
        st.info("At Q = 1, E = E°. Use a balanced redox reaction and a correctly defined reaction quotient for chemically meaningful results.")
        if st.button("Record electrochemistry calculation", key="record_nernst"):
            record_lab_activity("Virtual experiment", f"Nernst equation; E={potential:.4f} V")
            st.success("Calculation added to your session activity log.")

    with exp_tab4:
        st.subheader("Beer–Lambert law: absorbance and concentration")
        st.caption("Educational model: A = εlc. The solution assumes a linear response and a single absorbing species.")
        bl1, bl2, bl3 = st.columns(3)
        with bl1:
            epsilon = st.number_input("Molar absorptivity ε (L mol⁻¹ cm⁻¹)", min_value=0.001, max_value=1_000_000.0, value=150.0, step=10.0, key="beer_epsilon")
        with bl2:
            path_length = st.number_input("Path length l (cm)", min_value=0.01, max_value=20.0, value=1.0, step=0.1, key="beer_path")
        with bl3:
            concentration = st.number_input("Concentration c (mol/L)", min_value=0.0, max_value=10.0, value=0.002, step=0.001, format="%.4f", key="beer_conc")
        absorbance = beer_lambert_absorbance(epsilon, path_length, concentration)
        st.metric("Predicted absorbance A", f"{absorbance:.4f}")
        conc_from_abs = st.number_input("Or enter measured absorbance to estimate concentration", min_value=0.0, max_value=10.0, value=float(min(absorbance, 10.0)), step=0.01, key="beer_measured_abs")
        st.metric("Estimated concentration", f"{beer_lambert_concentration(conc_from_abs, epsilon, path_length):.6g} mol/L")
        beer_df = pd.DataFrame({"Concentration (mol/L)": np.linspace(0, max(concentration * 2, 0.001), 101)})
        beer_df["Predicted absorbance"] = epsilon * path_length * beer_df["Concentration (mol/L)"]
        fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(beer_df["Concentration (mol/L)"], beer_df["Predicted absorbance"]); ax.set_xlabel("Concentration (mol/L)"); ax.set_ylabel("Absorbance"); ax.set_title("Beer–Lambert calibration line"); ax.grid(True, alpha=0.25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.download_button("Download calibration data (CSV)", beer_df.to_csv(index=False).encode("utf-8"), "beer_lambert_calibration.csv", "text/csv")
        if st.button("Record Beer–Lambert experiment", key="record_beer_lambert"):
            record_lab_activity("Virtual experiment", f"Beer–Lambert law; predicted absorbance={absorbance:.4f}")
            st.success("Experiment saved to your learning record.")

    with exp_tab5:
        st.subheader("Arrhenius equation: temperature and rate constant")
        st.caption("Model: k = A exp(−Ea/RT). This illustrates how the rate constant changes with absolute temperature.")
        ar1, ar2, ar3 = st.columns(3)
        with ar1:
            activation_kj = st.number_input("Activation energy Ea (kJ/mol)", min_value=0.1, max_value=500.0, value=50.0, step=1.0, key="arr_ea")
        with ar2:
            pre_exponential = st.number_input("Pre-exponential factor A (s⁻¹)", min_value=0.000001, max_value=1e15, value=1e10, format="%.6g", key="arr_a")
        with ar3:
            temp_kelvin = st.number_input("Temperature (K)", min_value=100.0, max_value=2000.0, value=298.15, step=5.0, key="arr_t")
        gas_r = 8.314462618
        arr_k = arrhenius_rate_constant(pre_exponential, activation_kj, temp_kelvin)
        st.metric("Estimated rate constant k", f"{arr_k:.6g} s⁻¹")
        temperatures = np.linspace( max(100.0, temp_kelvin * 0.6), min(2000.0, temp_kelvin * 1.4), 150)
        arr_df = pd.DataFrame({"Temperature (K)": temperatures})
        arr_df["1/T (K⁻¹)"] = 1.0 / temperatures
        arr_df["ln(k)"] = math.log(pre_exponential) - (activation_kj * 1000.0) / (gas_r * temperatures)
        arr_df["k (s⁻¹)"] = np.exp(arr_df["ln(k)"].clip(lower=-700, upper=700))
        fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(arr_df["Temperature (K)"], arr_df["k (s⁻¹)"]); ax.set_xlabel("Temperature (K)"); ax.set_ylabel("Rate constant k (s⁻¹)"); ax.set_title("Arrhenius temperature dependence"); ax.grid(True, alpha=0.25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.download_button("Download Arrhenius data (CSV)", arr_df.to_csv(index=False).encode("utf-8"), "arrhenius_data.csv", "text/csv")
        if st.button("Record Arrhenius experiment", key="record_arrhenius"):
            record_lab_activity("Virtual experiment", f"Arrhenius equation; Ea={activation_kj:g} kJ/mol; T={temp_kelvin:g} K; k={arr_k:.6g} s^-1")
            st.success("Experiment saved to your learning record.")


# ============================================================
# UPGRADE MODULE 6: OFFLINE CHEMISTRY TUTOR
# ============================================================

elif page == "🤖 Chemistry Tutor":
    st.title("🤖 Chemistry Tutor")
    st.write("Ask a chemistry question for a guided explanation. The curated offline tutor always remains available, with optional live AI support when an API provider is configured.")
    tutor_topic = st.selectbox("Choose a topic", ["SMILES notation", "Molecular weight and descriptors", "Functional groups", "2D vs 3D structures", "First-order kinetics", "Acid–base titration", "Nernst equation", "Bonding and molecular geometry", "General question"])
    tutor_question = st.text_area("Your question", placeholder="Example: What does c1ccccc1 mean in SMILES?", height=100, key="tutor_question")
    tutor_guides = {
        "SMILES notation": "SMILES is a text notation for encoding molecular connectivity. Atoms are written as symbols, branches use parentheses, ring closures use matching digits, and bond symbols include = and #. Example: CCO represents ethanol; c1ccccc1 represents benzene. Validate a SMILES string with RDKit because syntax alone does not guarantee that a structure is chemically sensible.",
        "Molecular weight and descriptors": "Molecular weight is calculated from the atomic masses of the atoms in a molecule. RDKit can also calculate descriptors such as LogP (an estimate of lipophilicity), TPSA (topological polar surface area), hydrogen-bond donors/acceptors, and rotatable bonds. These are useful screening descriptors, not direct measurements of biological activity or solubility.",
        "Functional groups": "A functional group is a recognizable arrangement of atoms that contributes to a molecule's characteristic chemistry. Common examples include alcohols (R–OH), carboxylic acids (R–COOH), amines (R–NH₂), aldehydes (R–CHO), ketones (R–CO–R), esters (R–COO–R), and amides (R–CONH₂). Automated substructure matching is a guide and may have edge cases.",
        "2D vs 3D structures": "A 2D structure shows atom connectivity and bond types on a plane. A 3D conformer adds coordinates that represent one possible spatial arrangement. Flexible molecules may have several conformers, and a generated conformer is not necessarily the lowest-energy or experimentally observed structure.",
        "First-order kinetics": "For a first-order reaction, the rate is proportional to the concentration of reactant: rate = k[A]. The integrated law is ln([A]t/[A]0) = −kt, or [A]t = [A]0e^(−kt). The half-life is t₁/₂ = ln(2)/k and does not depend on initial concentration under the ideal first-order model.",
        "Acid–base titration": "In a strong monoprotic acid–strong base titration, the equivalence point occurs when acid and base have reacted in stoichiometric amounts. Before equivalence, excess acid determines pH; after equivalence, excess hydroxide determines pH. At 25 °C, the ideal strong acid–strong base equivalence point is approximately pH 7. Real measurements can differ because of activities, temperature, calibration, and other effects.",
        "Nernst equation": "The Nernst equation relates electrode potential to reaction conditions: E = E° − (RT/nF)ln(Q). E° is standard potential, R is the gas constant, T is temperature in kelvin, n is the number of electrons transferred, F is Faraday's constant, and Q is the reaction quotient. At 25 °C it is often written E = E° − (0.05916/n)log₁₀(Q).",
        "Bonding and molecular geometry": "Bonding and geometry depend on electron distribution and molecular structure. In VSEPR, electron domains around a central atom repel and arrange to reduce repulsion; lone pairs often compress adjacent bond angles. Hybridization labels are a useful introductory model, but they are not literal orbitals that can always be uniquely assigned to every atom.",
        "General question": "For a focused explanation, include the topic and the exact question. This offline tutor contains introductory guides for SMILES, molecular descriptors, functional groups, molecular visualization, kinetics, titration, electrochemistry, and molecular geometry."
    }
    ai_ready = ai_is_configured()
    if ai_ready:
        st.success("AI tutor is configured. Your question can be sent to the configured AI provider.")
        use_live_ai = st.toggle("Use live AI tutor for this answer", value=True, key="use_live_ai_tutor")
    else:
        use_live_ai = False
        st.info("Offline tutor mode is active. To enable AI answers, configure CHEM_LAB_AI_API_KEY in Streamlit secrets; never paste API keys into this page.")
    if st.button("Explain this topic", type="primary", key="ask_tutor"):
        answer = tutor_guides[tutor_topic]
        question_lower = tutor_question.lower().strip()
        # Match common keywords to a curated topic if no live AI is configured.
        keyword_map = [("smiles", "SMILES notation"), ("kinetic", "First-order kinetics"), ("half-life", "First-order kinetics"), ("titration", "Acid–base titration"), ("nernst", "Nernst equation"), ("functional group", "Functional groups"), ("logp", "Molecular weight and descriptors"), ("tpsa", "Molecular weight and descriptors"), ("3d", "2D vs 3D structures")]
        mapped_topic = tutor_topic
        for keyword, candidate_topic in keyword_map:
            if keyword in question_lower:
                mapped_topic = candidate_topic
                break
        if use_live_ai and tutor_question.strip():
            try:
                answer = ask_tutor(tutor_question.strip(), mapped_topic, "This is an undergraduate chemistry and cheminformatics virtual lab.")
                st.session_state.tutor_answer_source = "Configured AI provider"
            except Exception as exc:
                answer = tutor_guides.get(mapped_topic, tutor_guides["General question"])
                st.session_state.tutor_answer_source = "Offline fallback (AI request failed)"
                st.warning(f"AI response unavailable: {exc}. Showing the built-in guide instead.")
        else:
            answer = tutor_guides.get(mapped_topic, tutor_guides["General question"])
            if tutor_question.strip():
                answer += "\n\nYour question: " + tutor_question.strip() + "\n\nThis offline guide may not cover every detail. Add an AI provider key to get question-specific generated explanations."
            st.session_state.tutor_answer_source = "Built-in offline guide"
        st.session_state.tutor_last_topic = mapped_topic
        st.session_state.tutor_last_answer = answer
        record_lab_activity("Chemistry tutor", f"Topic: {mapped_topic}; source: {st.session_state.tutor_answer_source}")
    if st.session_state.get("tutor_last_answer"):
        st.subheader(f"Explanation: {st.session_state.get('tutor_last_topic', 'Chemistry')}")
        st.caption(f"Source: {st.session_state.get('tutor_answer_source', 'Built-in guide')}")
        st.markdown(st.session_state.tutor_last_answer)
        st.caption("Study tip: write the key equation, define each symbol, and solve one example to check your understanding.")
    st.warning("AI-generated explanations can be incorrect. Verify calculations and course-specific answers against your lecturer's notes and reliable references.")


# ============================================================
# UPGRADE MODULE 7: QUIZ AND STUDENT PROGRESS
# ============================================================

elif page == "🎯 Progress & Quiz":
    st.title("🎯 Student Progress & Practice Quiz")
    st.write("Complete a short quiz, review your activity during this session, and export a record for your own files.")
    quiz = [
        {"q": "Which SMILES string represents benzene in the common aromatic form?", "options": ["CCO", "c1ccccc1", "CC(=O)O", "O"], "answer": "c1ccccc1", "why": "Lowercase c represents aromatic carbon; the matching ring digit closes the six-membered ring."},
        {"q": "For an ideal first-order reaction, the half-life is…", "options": ["ln(2)/k", "k/2", "dependent on initial concentration only", "always 1 second"], "answer": "ln(2)/k", "why": "The integrated first-order rate law gives t₁/₂ = ln(2)/k."},
        {"q": "What does LogP primarily estimate?", "options": ["Number of atoms", "Lipophilicity", "Reaction time", "Cell voltage"], "answer": "Lipophilicity", "why": "LogP is the logarithm of a partition coefficient between octanol and water for a neutral species."},
        {"q": "At the equivalence point of an ideal strong acid–strong base titration at 25 °C, the pH is approximately…", "options": ["1", "4", "7", "14"], "answer": "7", "why": "At equivalence, the acid and base neutralize each other; the ideal solution is approximately neutral at 25 °C."},
        {"q": "In the Nernst equation, what does n represent?", "options": ["Temperature in °C", "Electrons transferred in the balanced redox reaction", "Molecular weight", "Number of atoms in the solvent"], "answer": "Electrons transferred in the balanced redox reaction", "why": "n is the stoichiometric number of electrons exchanged in the balanced redox process."},
        {"q": "Which statement about a generated 3D conformer is most accurate?", "options": ["It is always the experimentally observed structure", "It is one possible spatial arrangement", "It contains no bond information", "It proves biological activity"], "answer": "It is one possible spatial arrangement", "why": "Conformer generation proposes geometries; it does not guarantee an experimentally observed or global-minimum structure."}
    ]
    with st.form("chem_quiz_form"):
        quiz_name = st.text_input("Quiz attempt label (optional)", value="Practice attempt", key="quiz_attempt_label")
        answers = []
        for i, item in enumerate(quiz):
            st.markdown(f"**{i+1}. {item['q']}**")
            answers.append(st.radio("Choose an answer", item["options"], key=f"chem_quiz_{i}", index=None))
        submitted = st.form_submit_button("Submit quiz")
    if submitted:
        correct = sum(1 for i, item in enumerate(quiz) if answers[i] == item["answer"])
        score = round(100 * correct / len(quiz))
        review = [{"Question": item["q"], "Your answer": answers[i] or "Not answered", "Correct answer": item["answer"], "Result": "Correct" if answers[i] == item["answer"] else "Review", "Explanation": item["why"]} for i, item in enumerate(quiz)]
        attempt = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "student": st.session_state.get("student_name", "").strip() or "Student", "registration_number": st.session_state.get("registration_number", "").strip(), "label": quiz_name, "correct": correct, "total": len(quiz), "score_percent": score}
        st.session_state.chem_lab_quiz_scores.append(attempt)
        try:
            save_quiz_attempt(student_key, student_label, attempt, review)
            st.session_state.chem_lab_storage_error = ""
        except Exception as exc:
            st.session_state.chem_lab_storage_error = str(exc)
        record_lab_activity("Practice quiz", f"{correct}/{len(quiz)} correct ({score}%)")
        st.session_state.latest_quiz_review = review
        st.session_state.latest_quiz_attempt = attempt
    if st.session_state.get("latest_quiz_attempt"):
        attempt = st.session_state.latest_quiz_attempt
        st.subheader("Latest result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Score", f"{attempt['score_percent']}%")
        m2.metric("Correct", f"{attempt['correct']} / {attempt['total']}")
        m3.metric("Attempts this session", len(st.session_state.chem_lab_quiz_scores))
        st.dataframe(pd.DataFrame(st.session_state.latest_quiz_review), use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Persistent learning progress")
    try:
        summary = get_progress_summary(student_key)
        activity = get_activities(student_key)
        scores = get_quiz_attempts(student_key)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Activities recorded", summary["activity_count"])
        a2.metric("Quiz attempts", summary["quiz_count"])
        a3.metric("Best score", f"{summary['best_score']}%")
        a4.metric("Average score", f"{summary['average_score']}%")
        if activity:
            activity_df = pd.DataFrame(activity)
            st.dataframe(activity_df, use_container_width=True, hide_index=True)
            st.download_button("Export activity log (CSV)", activity_df.to_csv(index=False).encode("utf-8"), "chemistry_lab_activity.csv", "text/csv")
        else:
            st.info("No activities recorded yet. Record an experiment or ask the tutor a question to start your log.")
        if scores:
            scores_df = pd.DataFrame(scores)
            st.subheader("Quiz history")
            st.dataframe(scores_df, use_container_width=True, hide_index=True)
            st.download_button("Export quiz history (CSV)", scores_df.to_csv(index=False).encode("utf-8"), "chemistry_quiz_history.csv", "text/csv")
        export_bundle = {"student": student_label, "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"), "summary": summary, "activities": activity, "quiz_attempts": scores}
        st.download_button("Export all progress (JSON)", json.dumps(export_bundle, indent=2, ensure_ascii=False).encode("utf-8"), "chemistry_lab_progress.json", "application/json")
        st.caption("Progress is saved in a local SQLite database and can survive browser refreshes. A local database on Streamlit Community Cloud may be reset or unavailable across redeploys; use managed PostgreSQL or another persistent database for reliable multi-device accounts.")
        if st.session_state.get("chem_lab_storage_error"):
            st.warning(f"Could not save the latest activity to persistent storage: {st.session_state.chem_lab_storage_error}")
    except Exception as exc:
        st.error(f"Could not load persistent progress: {exc}")
        st.info("You can still export the session records as a backup.")
        activity = st.session_state.chem_lab_activity
        scores = st.session_state.chem_lab_quiz_scores
        export_bundle = {"student": student_label, "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"), "activities": activity, "quiz_attempts": scores}
        st.download_button("Export session progress (JSON)", json.dumps(export_bundle, indent=2, ensure_ascii=False).encode("utf-8"), "chemistry_lab_progress.json", "application/json")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧪 Cheminformatics Virtual Laboratory | "
    "Developed using Python, Streamlit and RDKit"
)
