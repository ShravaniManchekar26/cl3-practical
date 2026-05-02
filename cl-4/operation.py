# -------------------------------
# Fuzzy Set Operations
# -------------------------------

# Define fuzzy sets
A = {'x1': 0.2, 'x2': 0.5, 'x3': 0.7}
B = {'x1': 0.6, 'x2': 0.3, 'x3': 0.8}

# -------------------------------
# 1. UNION (max)
# -------------------------------
def fuzzy_union(A, B):
    return {x: max(A[x], B[x]) for x in A}

# -------------------------------
# 2. INTERSECTION (min)
# -------------------------------
def fuzzy_intersection(A, B):
    return {x: min(A[x], B[x]) for x in A}

# -------------------------------
# 3. COMPLEMENT (1 - μ)
# -------------------------------
def fuzzy_complement(A):
    return {x: round(1 - A[x], 2) for x in A}

# -------------------------------
# 4. DIFFERENCE (A - B)
# -------------------------------
def fuzzy_difference(A, B):
    return {x: round(min(A[x], 1 - B[x]), 2) for x in A}


# -------------------------------
# Fuzzy Relations
# -------------------------------

# Cartesian Product (A x B)
def cartesian_product(A, B):
    relation = {}
    for a in A:
        for b in B:
            relation[(a, b)] = min(A[a], B[b])
    return relation


# Max-Min Composition
def max_min_composition(R1, R2, X, Y, Z):
    result = {}
    for x in X:
        for z in Z:
            max_val = 0
            for y in Y:
                val = min(R1[(x, y)], R2[(y, z)])
                if val > max_val:
                    max_val = val
            result[(x, z)] = max_val
    return result


# -------------------------------
# Execution
# -------------------------------

print("Union:", fuzzy_union(A, B))
print("Intersection:", fuzzy_intersection(A, B))
print("Complement of A:", fuzzy_complement(A))
print("Difference (A - B):", fuzzy_difference(A, B))


# -------------------------------
# Define fuzzy sets for relations
# -------------------------------
A_rel = {'x1': 0.5, 'x2': 0.8}
B_rel = {'y1': 0.6, 'y2': 0.9}
C_rel = {'z1': 0.7, 'z2': 0.4}

# ✔ FIX: Automatically extract sets
X = list(A_rel.keys())
Y = list(B_rel.keys())
Z = list(C_rel.keys())

# -------------------------------
# Create Relations
# -------------------------------
R1 = cartesian_product(A_rel, B_rel)
R2 = cartesian_product(B_rel, C_rel)

print("\nRelation R1 (A x B):")
for k, v in R1.items():
    print(k, ":", v)

print("\nRelation R2 (B x C):")
for k, v in R2.items():
    print(k, ":", v)

# -------------------------------
# Max-Min Composition
# -------------------------------
composition = max_min_composition(R1, R2, X, Y, Z)

print("\nMax-Min Composition (R1 o R2):")
for k, v in composition.items():
    print(k, ":", v)