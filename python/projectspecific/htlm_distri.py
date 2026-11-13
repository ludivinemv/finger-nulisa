# Configuration

# Generate HTML
html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 15px;
    }
    figure {
      margin: 0;
      text-align: center;
    }
    figure img {
      width: 100%;
      height: auto;
      border-radius: 4px;
      object-fit: cover;
    }
    figcaption {
      margin-top: 5px;
      font-size: 0.9em;
      color: #555;
    }
  </style>
</head>
<body>

<h1>Distribution of baseline BBM.</h1>
<div class="gallery">
"""
