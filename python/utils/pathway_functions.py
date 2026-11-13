#! PUBLIC IMPORTS
from pathlib import Path
import pandas as pd
import numpy as np  

import requests
import networkx as nx
import community as community_louvain
from io import StringIO

import math

import seaborn as sns
import matplotlib.pyplot as plt

import re 
from urllib.parse import quote_plus
import gseapy as gp

from tqdm import tqdm 
from matplotlib import cm
from matplotlib.cm import ScalarMappable

from matplotlib.colors import Normalize

def get_string_ppi(proteins:list, species:int=9606, score:int=400):
    """Get PPI from Srting-db

    Args:
        proteins (list): list of proteins
        species (int, optional): _description_. Defaults to 9606.
        score (int, optional): _description_. Defaults to 400.

    """
    identifiers = "%0d".join(proteins)
    url = f"https://string-db.org/api/tsv/network?identifiers={identifiers}&species={species}&required_score={score}"
    r = requests.get(url)
    df = pd.read_csv(StringIO(r.text), sep="\t")
    return df

def get_pathway_url(term:str, gene_set:str) -> str:
    """ Generate a URL for a given pathway term based on the gene set database.

    Args:
        term (str): pathway term
        gene_set (str): gene set database name

    Returns:
        str: URL to the pathway information page
    """
    if gene_set == 'Reactome_2022':
        # Search Reactome pathway
        return f"https://reactome.org/content/query?q={quote_plus(term)}&species=Homo+sapiens "
    
    elif gene_set == 'KEGG_2021_Human':
        # KEGG IDs usually in format: "hsa04110 Cell Cycle"
        match = re.search(r"(hsa\d+)", term)
        if match:
            return f"https://www.kegg.jp/pathway/{match.group(1)} "
        else:
            return f"https://www.kegg.jp/kegg-bin/search_pathway_text?map=map&keyword={quote_plus(term)}&mode=1&viewImage=true"
    
    elif gene_set == 'WikiPathway_2021_Human':
        # Format: WP554~Autophagy
        match = re.search(r"(WP\d+)", term)
        if match:
            return f"https://www.wikipathways.org/pathways/{match.group(1)} "
    
    elif gene_set == 'GO_Biological_Process_2021':
        # Format: GO:0006355~regulation of transcription
        match = re.search(r"(GO:\d+)", term)
        if match:
            return f"https://amigo.geneontology.org/amigo/term/{match.group(1)} "
    
    return None  # Default fallback

def get_cluster(fdr_proteins, all_proteins, min_per_cluster = 2):
    """Get clusters

    Args:
        fdr_proteins (_type_): Most Significant proteins
        all_proteins (_type_): All significant proteins
        min_per_cluster (int, optional): Minimum number of protein per cluster. Defaults to 2.

    """
    ppi_df = get_string_ppi(all_proteins)

    G = nx.Graph()

    # Add nodes with size info
    for p in all_proteins:
        G.add_node(p, size=600 if p in fdr_proteins else 300)

    # Add edges
    for _, row in ppi_df.iterrows():
        G.add_edge(row['preferredName_A'], row['preferredName_B'])

    partition = community_louvain.best_partition(G)

    # Assign cluster info
    for node, cluster_id in partition.items():
        G.nodes[node]['cluster'] = cluster_id

    cluster_to_go = {}
    for cid in tqdm(set(partition.values())):
        cluster_genes = [n for n in G.nodes if G.nodes[n].get('cluster') == cid]
        
        if len(cluster_genes) < min_per_cluster:
            continue  # Skip very small clusters
        enr = gp.enrichr(gene_list=cluster_genes,
                        gene_sets=['KEGG_2021_Human',
                                    'Reactome_2022',
                                    'GO_Biological_Process_2021'],
                        organism='Human',
                        outdir=None,  # Don't write files
                        cutoff=0.05)
        
        if not enr.results.empty:
            top_term = enr.results.sort_values("Adjusted P-value").iloc[0]['Term']
            cluster_to_go[cid] = top_term

        print(cid, top_term)

    return partition, cluster_to_go, G

        
def get_network(path_options:Path|str, cluster_to_go, partition, G, study:str, radius:float = 1, angle:float = 3.2, num:int = 1):
    """Get PPI Network

    Args:
        path_options (Path | str): Path to save
        cluster_to_go (_type_): Clusters
        partition (_type_): Partition
        G (_type_): Graph
        study (str): Study version
        radius (float, optional): Radius option for visualisation. Defaults to 1.
        angle (float, optional): Angle option for visualisation. Defaults to 3.2.
        num (int, optional): Version of the network (in case we want to save several versions). Defaults to 1.
    """
    # Keep only nodes in enriched clusters 
    valid_cluster_ids = set(cluster_to_go.keys())
    filtered_nodes = [node for node in G.nodes() if partition.get(node) in valid_cluster_ids]

    # Create a subgraph with only enriched (colored) nodes
    G_filtered = G.subgraph(filtered_nodes).copy()

    # Rebuild partition and cluster groupings
    partition_filtered = {node: partition[node] for node in G_filtered.nodes()}
    clustered_nodes = {}
    for node, cid in partition_filtered.items():
        clustered_nodes.setdefault(cid, []).append(node)

    # Layout each cluster in a ring 
    pos = {}
    n_clusters = len(clustered_nodes)
    angle_step = angle * math.pi / n_clusters
    cluster_centers = {}

    unique_clusters = list(clustered_nodes.keys())
    #colors = plt.cm.tab20.colors  # 20 distinct colors
    colors = plt.cm.Set3.colors  # 12 colors

    cluster_color_map = {}
    for i, cid in enumerate(unique_clusters):
        cluster_color_map[cid] = colors[i % len(colors)]

    # Then when drawing nodes:

    for i, (cid, nodes) in enumerate(clustered_nodes.items()):
        angle = i * angle_step
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        cluster_centers[cid] = (cx, cy)

        subG = G_filtered.subgraph(nodes)
        local_pos = nx.spring_layout(subG, seed=42, k=1)

        for node, (x, y) in local_pos.items():
            pos[node] = (x + cx, y + cy)

    # Draw network
    plt.figure(figsize=(15, 12))

    # Draw edges
    nx.draw_networkx_edges(G_filtered, pos, alpha=0.3)

    # Draw nodes by cluster
    for node in G_filtered.nodes():
        size = G_filtered.nodes[node].get('size')
        cluster = partition_filtered[node]
        color = cluster_color_map[partition_filtered[node]]
        #color = color #f"C{cluster % 10}"
        
        nx.draw_networkx_nodes(G_filtered, pos,
                            nodelist=[node],
                            node_color=color,
                            node_size=size,
                            edgecolors='black',
                            linewidths=0.3)

    # Optional: Labels
    nx.draw_networkx_labels(G_filtered, pos, font_size=6)

    # === Step 4: Add GO term annotations ===
    for cid, label in cluster_to_go.items():
        cx, cy = cluster_centers.get(cid, (0, 0))
        plt.text(cx, cy + 0.5, label, fontsize=9, ha='center', weight='bold')

    plt.title("Clustered PPI Network (FDR-significant proteins only)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path_options/f'final_ppi_network_{study}_{num}.png', bbox_inches='tight', dpi=300, pad_inches = 0)
    plt.show()
    plt.close()

def get_pathway_figure(results:pd.DataFrame, 
                       path_options:Path|str, 
                       study:str, 
                       overlap:bool = True, 
                       x_col:str = 'Odds Ratio', 
                       color_col:str = 'Adjusted P-value', 
                       top_num:int = 10, 
                       top_col:str = 'Combined Score',
                       ):
    """Get figure with top pathways

    Args:
        results (pd.DataFrame): Results from the pathway analysis
        path_options (Path | str): Path to save
        study (str): Study version
        overlap (bool, optional): If True, show Overlap in parenthesis (number of proteins in the pathway). Defaults to True.
        x_col (str, optional): Value to consider for x. Defaults to 'Odds Ratio'.
        color_col (str, optional): Value to consider for colors. Defaults to 'Adjusted P-value'.
        top_num (int, optional): Number of pathway to print. Defaults to 10.
        top_col (str, optional): Value to consider for top pathways. Defaults to 'Combined Score'.
    """

    top = results.copy()
    if top_col == 'Combined Score':
        top = top.sort_values(top_col, ascending=False )
    else:
        top = top.sort_values(top_col, ascending=True )
    top = top.head(top_num)

    if overlap:
        y = top['Label']
    else:
        y = top['Term']

    pattern = r"\s?\(GO:\d+\)|\s?WP\d+|\s?R-HSA-\d+"
    y = y.str.replace(pattern, '', regex=True).str.strip()

    x = top[x_col]
    
    if color_col == 'Adjusted P-value':
        label_n = '-log10(FDR)'
        color_txt = 'log_fdr'
        log_pval = -np.log10(top['Adjusted P-value']) #top['Adjusted P-value']

        # Normalize log_pval for colormap (values between 0 and 1)
        norm = Normalize(vmin=log_pval.min(), vmax=log_pval.max())
        cmap = cm.get_cmap("vlag")
        colors = [cmap(norm(val)) for val in log_pval]
    else:
        label_n = color_col
        color_txt = color_col
        col_color = top[color_col]
        # Normalize log_pval for colormap (values between 0 and 1)
        norm = Normalize(vmin=col_color.min(), vmax=col_color.max())
        cmap = cm.get_cmap("vlag")
        colors = [cmap(norm(val)) for val in col_color]


    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=x,
        y=y,
        palette=colors,
        ax=ax
    )

    # Label axes and title
    ax.set_xlabel(x_col)
    ax.set_title(f'Top {top_num} of enriched pathways according\nto the {top_col}')
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required for ScalarMappable to work properly
    cbar = fig.colorbar(sm, ax=ax, label=label_n)

    #plt.tight_layout()
    plt.savefig(path_options/f'top_pathway_01_{study}_overlap_{overlap}_x_ {x_col}_color_{color_txt}_top_{top_col}.png', bbox_inches='tight')

    plt.show()

