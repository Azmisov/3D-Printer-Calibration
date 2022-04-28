import bmesh
from mathutils import Vector, Matrix

EPSILON = 1e-5

def normed_vector(vec, copy=False):
	if not isinstance(vec, Vector):
		vec = Vector(vec)
		vec.normalize()
		return vec
	if copy:
		return vec.normalized()
	vec.normalize()
	return vec

def fltr(bm, v=False, e=False, f=False):
	""" Filter list or bmesh object for vertices/edges/faces """
	if isinstance(bm, bmesh.types.BMesh):
		lst = []
		if v: lst += bm.verts[:]
		if e: lst += bm.edges[:]
		if f: lst += bm.faces[:]
		return lst
	# filter existing list
	types = []
	if v: types.append(bmesh.types.BMVert)
	if e: types.append(bmesh.types.BMEdge)
	if f: types.append(bmesh.types.BMFace)
	types = tuple(types)
	return list(filter(lambda x: isinstance(x, types), bm))

def connecting_geometry(verts, v=False, e=False, f=False):
	""" Get geometry that is connected to these verts. This is like
		region extend, but applied infinitely
	"""
	sel = set(verts)
	sel_new = sel
	# extend selection using connecting edges
	sel_pending = set()
	while len(sel_new):
		for vi in sel_new:
			for ei in vi.link_edges:
				for vj in ei.verts:
					if vj not in sel:
						sel_pending.add(vj)
		sel.update(sel_pending)
		sel_new = sel_pending
		sel_pending = set()
	if e or f:
		geo = set()
		for vi in sel:
			if e: geo.update(vi.link_edges)
			if f: geo.update(vi.link_faces)
		if v: geo.update(sel)
	else: geo = sel
	return list(geo)

def fill_edges_with_faces(bm, edges):
	""" Find groups of connected edges in `edges` and create faces from them;
		currently not very intelligent and won't look for "borders"
	"""
	# find edge "loops" (not BMLoop's) to create faces from
	loops = [] # [list(edges), ...]
	verts = {} # vert -> loop
	for e in edges:
		# loops this edge's verts have been found in
		l = []
		for v in e.verts:
			if v in verts:
				lst = verts[v]
				if lst not in l:
					l.append(lst)
		# two loops? they are now joined by this edge
		if len(l) > 1:
			l[0].extend(l[1])
			for e2 in l[1]:
				for v in e2.verts:
					verts[v] = l[0]
			loops.remove(l[1])
		# create new loop
		elif not len(l):
			lst = []
			l.append(lst)
			loops.append(lst)
		l[0].append(e)
		for v in e.verts:
			verts[v] = l[0]
	# turn loops into faces
	for l in loops:
		faces = bmesh.ops.contextual_create(bm, geom=l)["faces"]
		"""
		if tweak:
			for f in faces:
				f.normal_update()
				bmesh.ops.translate(bm, vec=f.normal*tweak, verts=f.verts)
		"""

def clip_plane(bm, co, no, *, geo=None, delete=0, rip=False, make_faces=False):
	""" clip or simply cut geometry along a plane
		:param co: clipping plane center
		:param no: normal vector of clipping plane
		:param geo: geometry to be clipped; None to clip full bmesh
		:param delete: 0/False = do not delete, 1/True = delete outer geo, 2 = delete inner geo
		:param rip: separate the outer/inner halves after bisection
		:param make_faces: create new faces along the bisection
	"""
	if geo is None:
		geo = fltr(bm, True, True, True)
	ret = bmesh.ops.bisect_plane(bm,
		geom=geo,
		dist=EPSILON,
		plane_co=co,
		plane_no=no,
		clear_outer=(delete and delete != 2),
		clear_inner=(delete == 2))
	do_rip = not delete and rip
	if make_faces or do_rip:
		edges = fltr(ret["geom_cut"], e=True)
		# rip only needed if the outer half isn't being deleted
		if do_rip:
			edges = bmesh.ops.split_edges(bm, edges=edges)["edges"]
		if make_faces:
			fill_edges_with_faces(bm, edges)			
	return ret

def clip_square(bm, width, height):
	""" clip mesh within wxh square, with corner at origin """
	e = 0
	clip_plane(bm, (e,e,0), (-1,0,0), delete=True)    
	clip_plane(bm, (e,e,0), (0,-1,0), delete=True)
	clip_plane(bm, (e,height-e,0), (0,1,0), delete=True)
	clip_plane(bm, (width-e,e,0), (1,0,0), delete=True)

def remove_doubles_faces_indiv(bm, distance=EPSILON):
	""" Removes doubles from individual faces """
	weld = {}
	for f in list(bm.faces):
		# already welded?
		verts = []
		for v in f.verts:
			if v in weld:
				v = weld[v]
			if v not in verts:
				verts.append(v)
		# no longer a face
		if len(verts) < 3:
			continue
		vi = 0
		# O(n^2), but likely smaller constant than bmesh.ops.remove_doubles
		while vi+1 < len(verts):
			a = verts[vi]
			vi += 1
			vj = vi
			while vj < len(verts):
				b = verts[vj]
				if (a.co-b.co).length < distance:
					weld[b] = a
					del verts[vj]
				else:
					vj += 1
	if len(weld):
		bmesh.ops.weld_verts(bm, targetmap=weld)