import base64
import io
import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

# CAD processing libraries
try:
    import ezdxf
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Polygon
    DXF_SUPPORT = True
except ImportError:
    DXF_SUPPORT = False

try:
    import trimesh
    STL_SUPPORT = True
except ImportError:
    STL_SUPPORT = False

try:
    import cadquery as cq
    STEP_SUPPORT = True
except ImportError:
    STEP_SUPPORT = False

# PDF processing libraries
try:
    import pdf2image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logger = logging.getLogger(__name__)

class FileProcessor:
    SUPPORTED_IMAGE_TYPES = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
    SUPPORTED_DOCUMENT_TYPES = ['pdf']
    SUPPORTED_CAD_TYPES = ['dxf', 'stl', 'ply', 'obj', 'step', 'stp', 'iges', 'igs', '3ds']

    @staticmethod
    def _encode_image_to_base64(image_file):
        """画像ファイルをbase64エンコード"""
        try:
            if hasattr(image_file, 'type') and image_file.type == "application/pdf":
                if not PDF_SUPPORT:
                    st.error("PDF処理にはpdf2imageライブラリが必要です")
                    return None
                images = pdf2image.convert_from_bytes(image_file.read())
                if images:
                    buffered = io.BytesIO()
                    images[0].save(buffered, format="PNG")
                    image_file.seek(0)
                    return base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                image_file.seek(0)
                image_bytes = image_file.read()
                image_file.seek(0)
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"画像base64エンコードエラー: {e}")
            st.error(f"画像の処理中にエラーが発生しました: {e}")
            return None

    @staticmethod
    def _process_dxf_file(dxf_file):
        """DXFファイルを処理して画像とメタデータを生成"""
        if not DXF_SUPPORT:
            return None, {"error": "DXF処理ライブラリが利用できません"}
        
        try:
            dxf_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as temp_file:
                temp_file.write(dxf_file.read())
                temp_file_path = temp_file.name
            dxf_file.seek(0)
            
            doc = ezdxf.readfile(temp_file_path)
            msp = doc.modelspace()
            
            entities_info = {
                'lines': [],
                'circles': [],
                'arcs': [],
                'texts': [],
                'dimensions': [],
                'blocks': []
            }
            
            for entity in msp:
                if entity.dxftype() == 'LINE':
                    entities_info['lines'].append({
                        'start': tuple(entity.dxf.start[:2]),
                        'end': tuple(entity.dxf.end[:2])
                    })
                elif entity.dxftype() == 'CIRCLE':
                    entities_info['circles'].append({
                        'center': tuple(entity.dxf.center[:2]),
                        'radius': entity.dxf.radius
                    })
                elif entity.dxftype() == 'TEXT':
                    entities_info['texts'].append({
                        'text': entity.dxf.text,
                        'position': tuple(entity.dxf.insert[:2])
                    })
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for line in entities_info['lines']:
                ax.plot([line['start'][0], line['end'][0]], 
                       [line['start'][1], line['end'][1]], 'b-', linewidth=1)
            
            for circle in entities_info['circles']:
                circle_patch = Circle(circle['center'], circle['radius'], 
                                    fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(circle_patch)
            
            for text in entities_info['texts']:
                ax.text(text['position'][0], text['position'][1], text['text'], 
                       fontsize=8, ha='left')
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"DXF Drawing: {dxf_file.name}")
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            os.unlink(temp_file_path)
            
            metadata = {
                'file_type': 'DXF',
                'total_entities': len(list(msp)),
                'lines_count': len(entities_info['lines']),
                'circles_count': len(entities_info['circles']),
                'texts_count': len(entities_info['texts']),
                'layers': [layer.dxf.name for layer in doc.layers],
                'drawing_units': doc.header.get('$INSUNITS', 'Unknown'),
                'entities_detail': entities_info
            }
            
            return image_base64, metadata
            
        except Exception as e:
            logger.error(f"DXF処理エラー: {e}")
            return None, {"error": f"DXF処理中にエラーが発生しました: {e}"}

    @staticmethod
    def _process_stl_file(stl_file):
        """STLファイルを処理して複数角度の画像を生成"""
        if not STL_SUPPORT:
            return None, {"error": "STL処理ライブラリが利用できません"}
        
        try:
            stl_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as temp_file:
                temp_file.write(stl_file.read())
                temp_file_path = temp_file.name
            stl_file.seek(0)
            
            mesh = trimesh.load_mesh(temp_file_path)
            
            metadata = {
                'file_type': 'STL',
                'vertices_count': len(mesh.vertices),
                'faces_count': len(mesh.faces),
                'volume': float(mesh.volume),
                'surface_area': float(mesh.area),
                'bounds': mesh.bounds.tolist(),
                'center_mass': mesh.center_mass.tolist(),
                'is_watertight': mesh.is_watertight,
                'is_valid': mesh.is_valid
            }
            
            angles = [(0, 0), (90, 0), (0, 90), (45, 45)]
            images = []
            
            for i, (azimuth, elevation) in enumerate(angles):
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                ax.plot_trisurf(mesh.vertices[:, 0], 
                               mesh.vertices[:, 1], 
                               mesh.vertices[:, 2],
                               triangles=mesh.faces, 
                               alpha=0.8, cmap='viridis')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f"STL Model - View {i+1} ({azimuth}°, {elevation}°)")
                ax.view_init(elev=elevation, azim=azimuth)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                images.append(image_base64)
                plt.close()
            
            os.unlink(temp_file_path)
            
            return images[0], {**metadata, 'additional_views': images[1:]}
            
        except Exception as e:
            logger.error(f"STL処理エラー: {e}")
            return None, {"error": f"STL処理中にエラーが発生しました: {e}"}

    @staticmethod
    def _process_step_file(step_file):
        """STEPファイルを処理（CadQuery使用）"""
        if not STEP_SUPPORT:
            return None, {"error": "STEP処理ライブラリ（CadQuery）が利用できません"}
        
        try:
            step_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
                temp_file.write(step_file.read())
                temp_file_path = temp_file.name
            step_file.seek(0)
            
            result = cq.importers.importStep(temp_file_path)
            
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as stl_temp:
                stl_temp_path = stl_temp.name
            
            cq.exporters.export(result, stl_temp_path)
            
            with open(stl_temp_path, 'rb') as stl_temp_file:
                image_base64, metadata = FileProcessor._process_stl_file(stl_temp_file)
            
            if metadata and 'error' not in metadata:
                metadata['file_type'] = 'STEP'
                metadata['original_format'] = 'STEP/STP'
            
            os.unlink(temp_file_path)
            os.unlink(stl_temp_path)
            
            return image_base64, metadata
            
        except Exception as e:
            logger.error(f"STEP処理エラー: {e}")
            return None, {"error": f"STEP処理中にエラーが発生しました: {e}"}

    @classmethod
    def process_file(cls, file):
        file_extension = Path(file.name).suffix.lower().replace('.', '')

        if file_extension in cls.SUPPORTED_IMAGE_TYPES:
            return cls._encode_image_to_base64(file), None
        elif file_extension in cls.SUPPORTED_DOCUMENT_TYPES:
            return cls._encode_image_to_base64(file), None # PDFs are treated as images
        elif file_extension == 'dxf':
            return cls._process_dxf_file(file)
        elif file_extension == 'stl':
            return cls._process_stl_file(file)
        elif file_extension in ['step', 'stp']:
            return cls._process_step_file(file)
        elif file_extension in ['ply', 'obj'] and STL_SUPPORT:
            return cls._process_stl_file(file)
        else:
            return None, {"error": f"未対応のファイル形式です: {file_extension}"}