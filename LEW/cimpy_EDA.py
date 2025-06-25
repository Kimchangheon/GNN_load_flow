import cimpy
from pathlib import Path
import pprint
import importlib.metadata  # For retrieving CIMpy version

# Display the current CIMpy version
print("CIMpy version:", importlib.metadata.version("cimpy"))  # e.g., 1.2.0

# 1. Collect all XML files from the LEW directory
xml_files = [str(file) for file in Path("Raw").glob("*.xml")]

# 2. Import the CIM model (choose appropriate CGMES version)
grid = cimpy.cim_import(xml_files, "cgmes_v2_4_15")  # or "cgmes_v3_0_0"

# 3. Access topology objects from the imported model
topology = grid["topology"]  # Dictionary: mRID ➜ CIM object

# 4. Print statistics for selected CIM classes
wanted_classes = [
    "TopologicalNode", "ConnectivityNode", "Terminal",
    "Substation", "VoltageLevel", "Breaker", "Disconnector",
    "PowerTransformer", "ACLineSegment", "GeneratingUnit", "EnergyConsumer"
]


counts = {
    cls: sum(obj.__class__.__name__ == cls for obj in topology.values())
    for cls in wanted_classes
}

# Pretty-print the results
pprint.pp(counts)
