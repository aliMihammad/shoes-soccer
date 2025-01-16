from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import ast
import json
import re

app = Flask(__name__)

# Load atau buat model Word2Vec (gunakan model yang telah Anda latih)
model = Word2Vec.load("word2vec_model.model")  # Pastikan model sudah tersimpan dengan nama ini

# Fungsi untuk menghasilkan vektor untuk atribut produk
def get_vector(word):
    return model.wv[word].tolist() if word in model.wv else [0] * 50


def get_chatbot_similarity (row, user_vectors):
    # Menghitung cosine similarity untuk vektor-vektor
    position_similarity = cosine_similarity([ast.literal_eval(row["position_vector"])], [user_vectors["position_vector"]])[0][0]
    surface_similarity = cosine_similarity([ast.literal_eval(row["surface_vector"])], [user_vectors["surface_vector"]])[0][0]
    brand_similarity = cosine_similarity([ast.literal_eval(row["brand_vector"])], [user_vectors["brand_vector"]])[0][0]
    material_similarity = cosine_similarity([ast.literal_eval(row['material_vector'])], [user_vectors["material_vector"]])[0][0]
    series_similarity = cosine_similarity([ast.literal_eval(row["series_vector"])], [user_vectors["series_vector"]])[0][0]
    color_similarity = cosine_similarity([ast.literal_eval(row["color_vector"])], [user_vectors["color_vector"]])[0][0]

    # Membuat dictionary dengan hasil cosine similarity dan ID
    similarities = {
        "shoes_id": row['shoes_id'],
        "brand_id": row['brand_id'],
        "material_id": row['material_id'],
        "position_id": row['position_id'],
        "series_id": row['series_id'],
        "surface_id": row['surface_id'],
        "color": row['color'],
        "position_similarity": position_similarity,
        "surface_similarity": surface_similarity,
        "brand_similarity": brand_similarity,
        "material_similarity": material_similarity,
        "series_similarity": series_similarity,
        "color_similarity": color_similarity
    }

    # Menghitung rata-rata dari similarity vektor (tanpa ID)
    vector_similarities = [position_similarity, surface_similarity, brand_similarity, material_similarity, series_similarity,color_similarity]
    similarities["average_similarity"] = np.mean(vector_similarities)
    
    return similarities

def get_search_similarity(row, user_vectors):
    # Menghitung cosine similarity untuk vektor-vektor
    color_similarity = cosine_similarity([ast.literal_eval(row["color_vector"])], [user_vectors["color_vector"]])[0][0]
    position_similarity = cosine_similarity([ast.literal_eval(row["position_vector"])], [user_vectors["position_vector"]])[0][0]
    surface_similarity = cosine_similarity([ast.literal_eval(row["surface_vector"])], [user_vectors["surface_vector"]])[0][0]
    brand_similarity = cosine_similarity([ast.literal_eval(row["brand_vector"])], [user_vectors["brand_vector"]])[0][0]
    material_similarity = cosine_similarity([ast.literal_eval(row['material_vector'])], [user_vectors["material_vector"]])[0][0]
    series_similarity = cosine_similarity([ast.literal_eval(row["series_vector"])], [user_vectors["series_vector"]])[0][0]

     # Membuat dictionary dengan hasil cosine similarity dan ID
    similarities = {
        "shoes_id": row['shoes_id'],
        "Nama Produk": row['shoes_name'],
        "color_similarity": color_similarity,
        "position_similarity": position_similarity,
        "surface_similarity": surface_similarity,
        "brand_similarity": brand_similarity,
        "material_similarity": material_similarity,
        "series_similarity": series_similarity
    }

    # Menghitung rata-rata dari similarity vektor (tanpa ID)
    vector_similarities = [color_similarity, position_similarity, surface_similarity, brand_similarity, material_similarity, series_similarity, color_similarity]
    similarities["average_similarity"] = np.mean(vector_similarities)
    
    return similarities

def process_input(user_input, keywords):
    # Normalisasi input (lowercase, hapus tanda baca)
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', '', user_input)  # Hapus tanda baca

    # Frasa yang perlu dicocokkan secara khusus
    special_color_keywords = ["merah muda", "biru muda"]

    # Pencocokan untuk kata kunci frasa terlebih dahulu
    extracted = {
        "color": None,
        "surface": None,
        "brand": None,
        "material": None,
        "series": None,
        "position": None,
        "price_range": None,
    }

    # Mencocokkan frasa khusus dan menambahkannya ke dalam extracted
    for phrase in special_color_keywords:
        if re.search(r'\b' + re.escape(phrase) + r'\b', user_input):
            extracted["color"] = phrase
            user_input = re.sub(r'\b' + re.escape(phrase) + r'\b', "", user_input)  # Hapus frasa dari input

    tokens = user_input.split()

    # Pencocokan dengan kata kunci
    for token in tokens:
        for key, values in keywords.items():
            if isinstance(values, dict):  # Jika keyword berupa dictionary (surface, position, material)
                if token in values:
                    extracted[key] = values[token].lower()  # Pastikan nilai menjadi lowercase
            elif isinstance(values, set):  # Jika keyword berupa set (brand, series, price)
                if token in values and key != "price":
                    extracted[key] = token.lower()

    if re.search(r'\b(di bawah|dibawah)\s(\d+)\b', user_input):
        harga = re.search(r'\b(di bawah|dibawah)\s(\d+)\b', user_input)
        if harga:
            extracted["price_range"] = {"max": int(harga.group(2))}

    elif re.search(r'\b(di atas|diatas)\s(\d+)\b', user_input):
        harga = re.search(r'\b(di atas|diatas)\s(\d+)\b', user_input)
        if harga:
            extracted["price_range"] = {"min": int(harga.group(2))}

    elif "murah" in user_input or "terjangkau" in user_input:
        extracted["price_range"] = {"max": 500000}

    elif "mahal" in user_input or "lebih mahal" in user_input:
        extracted["price_range"] = {"min": 500000}

    else:
      angka_harga = re.search(r'\b(\d+)\b', user_input)
      if angka_harga:
        harga_angka = int(angka_harga.group(1))
        extracted["price_range"] = {
            "min": harga_angka,
            "max": harga_angka + 99999
        }

    return extracted

@app.route('/generate_vectors', methods=['POST'])
def generate_vectors():
    data = request.json
    brand_vector = get_vector(data['brand'].lower()) 
    surface_vector = get_vector(data['surface'].lower()) 
    position_vector = get_vector(data['position'].lower())
    material_vector = get_vector(data['material'].lower())
    series_vector = get_vector(data['series'].lower())
    color_vector = get_vector(data['color'].lower())

    return jsonify({
        'brand_vector': brand_vector, 
        'surface_vector': surface_vector,
        'position_vector': position_vector,
        'material_vector': material_vector,
        'series_vector': series_vector,
        'color_vector': color_vector
    }), print('get vector berhasil')

@app.route('/get_similarity', methods=['POST'])
def get_similarity():
    data = request.json
    position = data['position'].lower()
    surface = data['surface'].lower()
    brand = data['brand'].lower()
    material = data['material'].lower()
    series = data['series'].lower()
    color = data['color'].lower()
    prioritas = data['prioritas']
    range_price = data['price_range']
    
    min_price = int(range_price.get("min", 0))  
    max_price = float(range_price.get("max", float('inf'))) 

    # Mendapatkan vektor untuk setiap data yang diterima
    position_vector = get_vector(position)
    surface_vector = get_vector(surface)
    brand_vector = get_vector(brand)
    material_vector = get_vector(material)
    series_vector = get_vector(series)
    color_vector = get_vector(color)
    
    # Buat dictionary yang berisi semua vektor
    user_vectors = {
        "position_vector": position_vector,
        "surface_vector": surface_vector,
        "brand_vector": brand_vector,
        "material_vector": material_vector,
        "series_vector": series_vector,
        "color_vector": color_vector,
        }


    # Ambil data produk dari database
    products = data['database']

    # Filter produk berdasarkan kondisi max dan min
    filtered_products = [
        product for product in products if min_price <= int(product['price']) < max_price
    ]

    # Hitung similarity antara data pengguna dan produk yang sudah difilter
    recommendations = []
    for product in filtered_products:
        similarities = get_chatbot_similarity(product, user_vectors)
        recommendations.append(similarities)

    # Urutkan berdasarkan average_similarity dan ambil 16 produk dengan similarity tertinggi
    recommendation = sorted(recommendations, key=lambda x: (x[prioritas], x["average_similarity"]), reverse=True)[:16]

    # return hasil rekomendasi
    return jsonify(recommendation),print("sukses chatbot")

@app.route('/get_search_recommendation', methods=['POST'])
def get_search_recommendation():
    keywords = {
        "color": {'putih', 'hijau', 'silver', 'merah', 'hitam', 'kuning', 'biru', 'ungu', 'oren', 'emas'},
        "brand": {"specs", "ortuseight", "mills"},
        "surface": {
            "alami": "fg",
            "buatan": "ag",
            "berlumpur": "sg",
            "basah": "sg",
            "fg": "fg",
            "sg": "sg",
            "ag": "ag",
            "tf": "tf"
        },
        "position": {
            "penyerang": "striker",
            "gelandang": "midfielder",
            "bek": "defender",
            "kiper": "goalkeeper",
            "striker": "striker",
            "midfielder": "midfielder",
            "defender": "defender",
            "goalkeeper": "goalkeeper",
            "winger": "winger",
            "sayap": "winger"
        },
        "material": {
            "rajutan": "knit",
            "kain": "knit",
            "knit": "knit",
            "sintetis": "synthetic",
            "buatan": "synthetic",
            "synthetic": "synthetic",
            "alami": "leather",
            "kulit": "leather",
            "leather" : "leather"
        },
        "series": {
            "cepat": "kecepatan",
            "kecepatan": "kecepatan",
            "fleksibel": "fleksibel",
            "fleksibilitas": "fleksibel",
            "kontrol": "kontrol",
            "ringan": "ringan",
            "accelerator" : "fleksibel",
            "alphaform" : "fleksibel",
            'ls': 'kontrol',
            'spectra': 'kontrol',
            'lightspeed': 'kecepatan',
            'hyperspeed': 'kecepatan',
            'aerodyne': 'fleksibel',
            'alphaform': 'fleksibel',
            'preface': 'kecepatan',
            'speedkraft ': 'kontrol',
            'nightshade': 'kecepatan',
            'lightspeed': 'kecepatan',
            'reacto': 'fleksibel',
            'xyclops': 'fleksibel',
            'astro': 'kontrol',
            'gladion': 'fleksibel',
            'triton': 'fleksibel',
            'espada': 'fleksibel',
            'genome': 'kecepatan',
            'triton': 'kecepatan',
            'evos+': 'fleksibel',
            'sirius': 'fleksibel',
            'forte': 'fleksibel',
            'vantage': 'kecepatan',
            'catalyst': 'kecepatan',
            'legion': 'kecepatan',
            'vision': 'kecepatan',
            'olympico': 'kecepatan',
            'weave': 'fleksibel',
            'tiburon ': 'fleksibel',
            'volt': 'fleksibel',
            'nike': 'kecepatan'
        },
        "price": {"murah", "di bawah", "di atas", 'diatas', 'dibawah', 'terjangkau'},
    }



    data = request.json
    search_input = data['search_input']
    search_input_extracted = process_input(search_input, keywords)
    is_price_only = search_input_extracted.get("price_range") is not None and all(value is None for key, value in search_input_extracted.items() if key != "price_range")
    is_brand_only = search_input_extracted.get("brand") is not None and all(value is None for key, value in search_input_extracted.items() if key != "brand") or search_input_extracted.get("brand") is not None and search_input_extracted.get("price_range") is not None and all(value is None for key, value in search_input_extracted.items() if key not in ["brand", "price_range"])
    is_color_only = search_input_extracted.get("color") is not None and all(value is None for key, value in search_input_extracted.items() if key != "color") or search_input_extracted.get("color") is not None and search_input_extracted.get("price_range") is not None and all(value is None for key, value in search_input_extracted.items() if key not in ["color", "price_range"])
    is_surface_only = search_input_extracted.get("surface") is not None and all(value is None for key, value in search_input_extracted.items() if key != "surface") or search_input_extracted.get("surface") is not None and search_input_extracted.get("price_range") is not None and all(value is None for key, value in search_input_extracted.items() if key not in ["surface", "price_range"])
    is_material_only = search_input_extracted.get("material") is not None and all(value is None for key, value in search_input_extracted.items() if key != "material") or search_input_extracted.get("material") is not None and search_input_extracted.get("price_range") is not None and all(value is None for key, value in search_input_extracted.items() if key not in ["material", "price_range"])
    is_series_only = search_input_extracted.get("series") is not None and all(value is None for key, value in search_input_extracted.items() if key != "series") or search_input_extracted.get("series") is not None and search_input_extracted.get("price_range") is not None and all(value is None for key, value in search_input_extracted.items() if key not in ["series", "price_range"])
    is_position_only = search_input_extracted.get("position") is not None and all(value is None for key, value in search_input_extracted.items() if key != "position") or search_input_extracted.get("position") is not None and search_input_extracted.get("price_range") is not None and all(value is None for key, value in search_input_extracted.items() if key not in ["position", "price_range"])
    range_price = search_input_extracted['price_range']

    min_price = int(range_price.get("min", 0)) if range_price else 0
    max_price = float(range_price.get("max", float('inf'))) if range_price else float('inf')

    
    
    print("Input Search:", search_input)
    print("Extracted Data:", search_input_extracted)
    print('min_price:', min_price)
    print('max_price:', max_price)
    print("Is Price Only:", is_price_only)
    print("Is Brand Only:", is_brand_only)
    print("Is Color Only:", is_color_only)
    print("Is Surface Only:", is_surface_only)
    print("Is Material Only:", is_material_only)
    print("Is Series Only:", is_series_only)
    print("Is Position Only:", is_position_only)
    
    search_input_vector = {
        "position_vector": get_vector(search_input_extracted["position"]),
        "surface_vector": get_vector(search_input_extracted["surface"]),
        "brand_vector": get_vector(search_input_extracted["brand"]),
        "material_vector": get_vector(search_input_extracted["material"]),
        "series_vector": get_vector(search_input_extracted["series"]),
        "color_vector": get_vector(search_input_extracted["color"]),
    }

    products = data['database']
    
    filtered_products = [
        product for product in products if min_price <= int(product['price']) < max_price
    ]
    # return print(search_input_vector['position_vector'])
    
    recommendations = []

    if is_price_only:
        for product in filtered_products:
            similarities = get_search_similarity(product, search_input_vector)
            recommendations.append(similarities)

    elif is_brand_only:
        for product in filtered_products:
            similarities = get_search_similarity(product, search_input_vector)
            if np.isclose(similarities["brand_similarity"], 1.0):
                recommendations.append(similarities)
    elif is_color_only:
        for product in filtered_products:
            similarities = get_search_similarity(product, search_input_vector)
            if np.isclose(similarities["color_similarity"], 1.0):
                recommendations.append(similarities)

    elif is_surface_only:
        for product in filtered_products:
            similarities = get_search_similarity(product, search_input_vector)
            if np.isclose(similarities["surface_similarity"], 1.0):
                recommendations.append(similarities)

    elif is_material_only:
        for product in filtered_products:
            similarities = get_search_similarity(product, search_input_vector)
            if np.isclose(similarities["material_similarity"], 1.0):
                recommendations.append(similarities)

    elif is_series_only:
        for product in filtered_products:
            similarities = get_search_similarity(product, search_input_vector)
            if np.isclose(similarities["series_similarity"], 1.0):
                recommendations.append(similarities)

    elif is_position_only:
        for product in filtered_products:
            similarities = get_search_similarity(product, search_input_vector)
            if np.isclose(similarities["position_similarity"], 1.0):
                recommendations.append(similarities)

    else:
        for product in filtered_products:
            similarities = get_search_similarity(product, search_input_vector)
            if similarities["average_similarity"] > 0:
                recommendations.append(similarities)

        # Sorting recommendations by 'average_similarity' after the loop
        recommendations = sorted(
            recommendations, 
            key=lambda x: (
                0 if x['color_similarity'] < 1 else x['color_similarity'],  
                x["average_similarity"]
            ), 
            reverse=True
        )[:50]
    

    # Kembalikan hasil similarity
    return jsonify(recommendations),print("sukses")


# if __name__ == '__main__':
#     app.run(port=5000)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
