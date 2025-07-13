<p align="center">
  <img src="docs/images/PackMate Rendering 1.jpeg" alt="PackMate Rendering" width="600"/>
</p>

# PackMate - Nachhaltige Innenverpackung neu gedacht / Rethinking Sustainable Inner Packaging

[🇩🇪 Deutsch](#-überblick) | [🇬🇧 English](#-overview)

---

## Überblick

**PackMate** ist ein interdisziplinär entwickelter Prototyp entstanden im Rahmen der *Cross Innovation Class 2025*. Das Ziel: Kleinteile sicher und nachhaltig verpacken - ohne Luftpolsterfolie, dafür mit Karton. Das Gerät scannt Objekte mit zwei USB-Kameras, erstellt ein 3D-Modell und bereitet automatisch passendes Verpackungsmaterial (bestehend aus alten Pappkartons) vor.

<p align="center">
  <img src="docs/images/Packaging material in box final.jpeg" alt="Verpackungsmaterial Anwendung" width="500"/>
</p>

### Hauptfunktionen
- **3D-Scan** mit zwei Kameras
- **Maskenerkennung** und **STL-Modellgenerierung**
- **Lasercut-Vorbereitung** für Kartonverpackung
- Touchscreen-Interface (Pygame)
- Modularer Aufbau: Scan-, Lager- und Schneideeinheit

### Benutzeroberfläche

<p align="center">
  <img src="docs/images/UI concept.png" alt="UI-Konzept" width="600"/>
</p>

Die grafische Oberfläche wurde in Zusammenarbeit mit Designstudierenden der AMD Hamburg erstellt und ist vollständig über den Touchscreen steuerbar. 

### Projektstruktur

    .
    ├── docs/
    │   ├── images/  → Alle Produkt- und Konzeptbilder
    │   ├── pdfs/
    │   │   ├── PackMate - Entwicklerdokumentation.pdf
    │   │   └── PackMate Pitch (blurred).pdf
    ├── ui elements/ → UI-Grafiken
    ├── main.py      → Python-Skript für GUI, 3D-Scan, STL, etc.
    ├── README.md    → Diese Datei

### Dokumentation
- [PackMate Entwicklerdokumentation](docs/pdfs/PackMate%20-%20Entwicklerdokumentation.pdf)
- [PackMate Pitch Deck](docs/pdfs/PackMate%20Pitch%20(blurred).pdf)

---

## Overview

**PackMate** is a prototype developed in the *Cross Innovation Class 2025* by a multidisciplinary team. Its mission: help suppliers package small parts safely and sustainably - not with bubble wrap, but custom-fit, laser-cut cardboard.

<p align="center">
  <img src="docs/images/Packaging material in box final.jpeg" alt="Application packaging material" width="500"/>
</p>

### Features
- **3D scanning** using two cameras
- **Binary mask** generation and **STL 3D model** creation
- **Laser-cut packaging** design from cardboard
- Full **touchscreen UI** built with Pygame
- Modular hardware design: scanning, storage, and cutting units

### User Interface

<p align="center">
  <img src="docs/images/UI concept.png" alt="UI Concept" width="600"/>
</p>

The GUI was co-designed with industrial design students from AMD Hamburg for ease of use and professional appearance.

### Project Structure

    .
    ├── docs/
    │   ├── images/  → All product and concept images
    │   ├── pdfs/
    │   │   ├── PackMate - Entwicklerdokumentation.pdf
    │   │   └── PackMate Pitch (blurred).pdf
    ├── ui elements/ → UI graphics
    ├── main.py      → Python script for GUI, 3D scan, STL etc.
    ├── README.md    → This file

### Documentation
- [PackMate Developer Documentation](docs/pdfs/PackMate%20-%20Entwicklerdokumentation.pdf)
- [PackMate Pitch Deck](docs/pdfs/PackMate%20Pitch%20(blurred).pdf)

---

## Contributing

Pull Requests welcome! Feature suggestions or bug reports can be submitted via [GitHub Issues](https://github.com/kraemerlukas314/Cross-Innovation-Class-2025/issues).

---

## Credits

Developed by:
Lukas Krämer (FH Wedel), Marie Dittrich (Leuphana University Lüneburg), Kat Heitbaum (Brand University), Nikki Völz (AMD Hamburg), Yuri Gwak (AMD Hamburg), Jule Pfister (Leuphana University Lüneburg)

---

## License

MIT License

