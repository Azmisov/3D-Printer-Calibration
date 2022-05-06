"""
Checkerboard pattern;

Clean edges when checkerboard is folded at an angle:
----------------------------------------------------
I want to have the pattern on different angled faces. For edges, just need
to make sure when we cut away the pattern at the edge we don't leave a square
unsupported/disconnected from the rest of the material. There will also be
cases where part of a square gets cut off by one of the adjacent angled faces,
and so while not disconnected, it is now supported by less material. I'm inclined
to say every square should be given at least width*height support.

Things I've tried:
- Plain inset/extrude do not give what we want on edges; the direction of depth is skewompous
- Extrude each angled face individually, then bridge the bordering extrusions on the edge.
	This leaves proper support to each square, but the edge is stretched, so looks strange.
	Surely not what we want.
- Variant on previous, you can fill the edge triangle portion fully, and then it is like
	a beveled corner.
- Since previous just fills in the corner, might as well just have various disconnected
	faces, with a space between. Given pattern cutout depth D, we want to find offset from
	edge O which allows at least S width of support for the edge. This comes out to be:
		solve S = 2*sqrt(O^2 + D^2)*sin(67.5-arctan(D/O)) for O
		simpler formulation that is equivalent:
			O = D*(sqrt(2)-1) + S*sqrt(1-sqrt(2)/2)
	for a 45 degree angled surface.
- Previous two options give sort of a "safe" solution, but doesn't fully make use of the corner space.
	I'd imagine you could have some partial squares still on the corner, where the boolean
	cutout has been restricted. Here is the general case. At the edge, looking at a profile view,
	here are the vertices:
		p1 (D_surf1,0) -> end of cutout, surface 1
		p2 (0,0) -> start of cutout, surface 1
		p3 (0,O_surf1) -> edge of surface 1/2
		p4 (cos(45)*O_surf2, O_surf1+sin(45)*O_surf2) -> start of cutout, surface 2
		p5 (cos(45)*O_surf2+sin(45)*D_surf2, O_surf1+sin(45)*O_surf2-cos(45)*D_surf2) -> end of cutout, surface 2
	We want distance between first/last points to be >= S. 
	We'd probably also constrain the end of cutout of surface 2's to not go below the cut
	start. So the breaking surface area actually may increase in that case (the cutout end
	is further back than the start, so distance to the end may be greater). That will increase
	the x position by (O_surf1+O_surf2)*sqrt(2). Similar situation and constraints could be
	applied to surface 1 (and probably should be).
	
	So you can decrease the D for each surface and it will increase surface area. Setting
	both to zero doesn't give you S. However, experimenting setting the D of the smaller
	O to zero appears to work. In fact, the calculation is symmetric about the edge bisection,
	so we can just deal with the case where O_surf1 >= O_surf2, and the reverse is just the
	symmetric/mirrored case.
	
	My idea is we can form a plane normal to the surface, and use that as a limit for
	how far the other surface's depth can be. That will give a clean plane across the whole
	edge so will look nicer I think than adaptively modifying depth per-cutout. We may need
	to increase the angle (so greater than normal), if certain distributions of O_surf1/2
	do not provide enough S. Let's check it.
		p1 clamped = (min(D_surf1,O_surf1),0)
		p5 clamped = (cos(45)*O_surf2+sin(45)*min(D_surf2,O_surf2), O_surf1+sin(45)*O_surf2-cos(45)*min(D_surf2,O_surf2))
	Note cos(45) = sin(45).
	Seems this does not work for O_surf1=O_surf2, which is where surface area is minimal. So finding an
	angle that works for that case will sufficient to give proper surface area for all other cases.
	For depth zero, S will equal O*sqrt(2(1-cos(135)) via law of cosines = O*1.848. So even if depth is
	zero, we can't give enough surface area (would need to equal 2). 
	
	That tells us there is no such plane angle that can satisfy our constraint that S = 2*O. An
	alternative is to consider the angles at O_surf1 and O_surf2; normally the cutout angles would
	be 0deg and -45deg respectively; we can increase those angles to meet the S requirement. When
	angle equals 67.5-90=-22.5deg for both, then the surface area is kept constant at O*1.848 the whole
	way through. We can decrase (for surf1) and increase (for surf2) angle and that widens it at it's
	largest point, making attachment >= S. We want to find the minimal angle such that it = S.
		We're basically cutting the cutout along an angle, and want to find that point that it is narrowest
		(or deepest cut point). That turns out to be simple, and is just a simple change in the formula.
		Where before w/ p5, we walked along O_surf2/sqrt(2), here we walk an additional O' distance:
			O' = depth*tan(theta), where theta = 45-(22.5-t) = 22.5+t
		For p1, we do similarly, and y value becomes:
			O' = -depth*tan(theta), where theta also = 22.5+t
		So all together:
			(where O_surf1=O_surf2=o, D_surf1=D_surf2=d, sin(45)=cos(45)=1/sqrt(2), S=2*o)
			p1' (d, -d*tan(22.5+t))
			p5' ((o+d+d*tan(22.5+t))/sqrt(2), o + (o-d+d*tan(22.5+t))/sqrt(2))
			solve 2*o = sqrt(((o+d+d*tan(22.5+t)/sqrt(2) - d)^2 + (o + (o-d+d*tan(22.5+t))/sqrt(2) + d*tan(22.5+t))^2) for t
			a bit messy to solve, but ends up being a fairly clean quadratic solution with just a few constants

	If > 45 (if that is even possible), then this solution will not work, since p1'/p5' intersect the side wall of the cutout, instead of the backface.
	But I tested for the ranges of o/d I'm planning on using, and that doesn't appear to be an issue.

	So this can be viewed as an optimization of the previous bullet. It still has a sort of "border" on the edge, but we've
	reduced it's size by adding this angled cutoff. Border = 0.5*width, where before it may have been 3*width
	
	You could further optimize this and do a per-column cutoff plane, specializing for when O_surf1 != O_surf2. I won't
	do this because I think having the same cutoff plane for all columns will look cleaner (I imagine); also, this method
	is better for when we do the surface that is rotated 45 degree around Z-axis; in that case, O_surf1 and O_surf2 are
	sort of arbitrary, unless you decided to wrap around in that direction (but there, you'd still have a pattern disconnect
	on the 45*45 rotated corner piece). So I think in those cases you'd want to enforce a border of 0.5*width, and that is
	handled nicely by this method.

- Per column cutoff. So I did implement the previous option, but I didn't realize how much it would cut off for the larger
	widths. I think I will still use the previous for doing the vertical edge (though perhaps clamped so surface area only
	needs to be like .3 or something). But I will try a per-column cutoff so that 
	
	Good news is that the per column is essentially the same calculation as for all, except O_surf is independent now:
		p1' (d, -d*tan(theta))
		p5' ((o2+d+d*tan(theta))/sqrt(2), o1 + (o2-d+d*tan(theta))/sqrt(2))
	All that changed is we labeled o as either o1 or o2.
	
	For each column, two squares crossing the edge is possible. For three to cross, angle would need to be > atan(1/3),
	which is about 71deg; I'm not going that high, so no need to deal with that. In any case, I think we'd handle each
	pair of crossing squares individually so I don't think it matters. So one extra thing we'd need to deal with is the
	skewed squares. In this case, the surface area cross section changes across the width of the square; the angle cutoff
	is quadratic, with a minimum when |o1-o2| is minimal. The distance (depth*tan(theta)) the cutout is offset is also
	quadratic-like (perhaps not fully, but with the small angle changes I guess distance vs theta is linear). I think it will look
	cleaner if we don't do that curvature and instead just use the largest angle across
	the width for the whole thing.
	
	Another idea is that the smaller of o1/o2 should have a larger cutoff than the other. You'd imagine if one cutoff was
	right next to the edge (o1=0), the other is far away and shouldn't have to trim itself to compensate. At least visually,
	I think that would look nicer. For problem statement, it is a simple change:
		o1 = p*o
		o2 = (1-p)*o
		p1' (d, -d*tan(theta+(1-p)*delta))
		p5' (((1-p)*o+d+d*tan(theta+p*delta))/sqrt(2), p*o + ((1-p)*o-d+d*tan(theta+p*delta))/sqrt(2))
	I think the calculation will be simpler if we drop the angle parameterization and switch to just a plain
	distance shift; e.g. we decrease the width of the cutout away from the edge. We can convert back to angles later:
		t = amount shifted total
		pm1 (d, -(1-p)*t)
		pm5 (((1-p)*o + d + p*t)/sqrt(2), p*o + ((1-p)*o - d + p*t)/sqrt(2))
	Another point this brings up is whether pm5 can be below pm1 in y axis? That would make the current location a local
	minima and cause some other clipping issues that need to be taken care of. When p=.5, it's when depth > o*.5*tan(67.5).
	For other values of p it will need to be greater. I actually don't think this is a problem though, if we restrict t >= 0.
	At what depth does this solution become invalid, e.g. the angle > 45 case described in previous bullet? Calculating
	it in Sage, it turns out to be 2.31475737868327, or 1.10765059749672 if you say that when p=1, shift for surf1 must be zero.
	
	For there to be a linear solution, or constant, t would have depend linearly on p. I tried t = 1-p and it does
	actually work, giving constant surface area; but area is greater than it needs to be. Also thinking about it more, I
	do imagine having a constant shift will look better than one that is linear (in this case) or nonlinear. Having a constant
	shift I think won't conflict with the checkerboard shape as much.
	- d/dt is positive linear for both surf1 and surf2
	- d/dp is quadratic
	Since d/dt is linear, no need to do some nonlinear weighting on (1-p)*t or p*t.
	
	Okay so came up with a better formulation of the problem and that made a solution clear. The "inset" surface is given by
	the convex hull inset by the specified depth. The corner edge of this inset surface is 22.5deg from the outside corner,
	or D = depth*tan(22.5) distance on either side of the outside corner. When a checkerboard square goes within that distance
	to the edge, it needs to start rotating around that inner corner. Formally, given square is on one side of edge, shift
	it away from the edge by: max(0,depth*tan(22.5)-distance). For arbitrary fold angle (rotation from vertical) it would
	be max(0,depth*tan(.5*theta))
	
	For a square that is partially on one side or the other, the attaching surface area begins to decrease as it wraps around
	the corner. Here we use a similar solution to what we were doing all along. Surface area is distance between:
		p1 (0, 0)
		p2 ((1-p)*w/sqrt(2), p*w + (1-p)*w/sqrt(2))
		and we want to introduce a small shift to increase surface area via:
			p1 shift = (1-p)*t
			p2 shift = p*t
		e.g., preserve that 22.5 angle more for the side the square happens to be more on
		p1 (0, -(1-p)*t)
		p2 (((1-p)*w + p*t)/sqrt(2), p*w + ((1-p)*w + p*t)/sqrt(2))
	Solution to that is actually very clean.
	For arbitrary angle, just a slight change:
		p1: (0, -(1-p)*t)
		p2: (((1-p)*w + p*t)*cos(theta), p*w + ((1-p)*w + p*t)*sin(theta))
	
		
		
"""
import helpers as H
import bpy, bmesh, math
from collections import defaultdict
from mathutils import Vector, Matrix

EPSILON = H.EPSILON

def skewed_square(bm, rad, angle, offset=None):
	""" generates a skewed square w/ bottom-left corner at offset, returning coordinates of upper-right corner """
	angle *= math.pi/180.0
	sy = math.tan(angle)
	transform = Matrix.Shear("YZ",4,(sy,0)) @ Matrix.Translation((rad,rad,0.0))
	if offset:
		transform = Matrix.Translation(offset) @ transform
	bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=rad, matrix=transform)
	return Vector((1, sy+1, 0))*2*rad

def fold_edge_cutoff_angle(depth, radius1, radius2=None):
	""" Find optimal angle to allow geometry at a folded edge at least (2*radius)^2 surface
		area connecting it to the rest of the object. This cutoff angle is used to restrict
		the boolean subtraction. E.g. a plane starting at radius distance from edge of surface,
		with vector direction given by normal vector rotated cutoff_angle degrees away from edge.
		See pydocs above
	"""
	if radius2 is None:
		radius2 = radius1
	# terms of equation
	dd = depth**2
	oo = radius1*radius2
	do = depth*(radius1+radius2)
	a = 6.82842712474619*dd
	b = 6.82842712474619*do - 5.656854249492381*dd
	c = 1.1715728752538097*(dd-oo) - 2.8284271247461903*do
	# quadratic formula for tan(theta)
	tan_angle = (math.sqrt(b**2 - 4*a*c) - b)/(2*a)
	# and return angle
	return math.atan(tan_angle)

def fold_edge_clip(bm, geo, edge_coordinate, edge_tangent, surface_normal, radius, depth):
	""" Clip edge of a folded boolean subtract geometry, so that geometry on edge has
		minimum required attaching surface area. See pydocs above
		:param edge_coordinate: location of edge we're clipping against
		:param edge_tangent: direction of edge we're clipping against, where direction goes clockwise
			around the surface
		:param surface_normal: normal, point to the outside of the surface (we clip against inside)
	"""
	edge_tangent = H.normed_vector(edge_tangent)
	no = H.normed_vector(surface_normal)
	co = Vector(edge_coordinate) + radius*no.cross(edge_tangent)
	rads = fold_edge_cutoff_angle(depth, radius)
	no.rotate(Matrix.Rotation(rads-math.pi/2, 4, edge_tangent))
	H.clip_plane(bm, co, no, delete=2, geo=geo, make_faces=True)
	
def fold_edge_shift(w, d, p):
	""" Similar to fold_edge_cutoff_angle, but parameterized as a shift, and the shift
		can now be different for surf1 and surf2
	"""
	p2p = p^2-p
	a = w*(-1.1715728752538097*p2p - 2)
	b = w*(1.1715728752538097*p2p - 1.4142135623730951) + d*1.4142135623730951
	r = (
		d*d*(-1.3725830020304777*p2p - 0.3431457505076194) +
		w*w*(-5.656854249492381*p2p + 2) +
		d*w*(6.627416997969522*p2p + 1.6568542494923806)
	)
	return (-b - math.sqrt(r))/a

def border_angle(b, d, angle):
	""" Angle in radians to rotate border vector from normal, away from edge,
		to guarantee 2*b surface area attaching border
		:param b: border radius
		:param d: depth of checkerboard extrusion
		:param angle: angle of edge
	"""
	angle *= math.pi/180
	c = math.cos(angle)
	s = math.sin(angle)
	c2 = c*c
	s2 = s*s
	x = c2+s2+2*c+1
	radical = -d*d*(c2*(c2-2) + s2*(s2-2) + 2*c2*s2 + 1) + 4*b*b*x
	t = (-b*x + 2*d*s + math.sqrt(radical))/(d*x)
	return math.atan(t)

def create_checkerboard(cellsize, width, height, depth, width_cols, gap=5e-3):
	""" Creates a checkerboard, with columns skewed at various angles
		:param cellsize: width of individual squares
		:param width: width of checkerboard
		:param height: height of checkerboard
		:param depth: depth to extrude
		:param width_cols: width used to calculate skewing columns; beyond this width,
			we cycle backwards through the angles
		:param gap: if square tips align perfectly, boolean ops will give non-manifold results;
			so this adds a small offset so that it will be manifold
	"""
	# TODO: zigzag disabled for now; folding assumes skew angle is always positive
	zigzag = False
	
	bm = bmesh.new()
	
	rad = .5*cellsize
	goffset = 2*cellsize+gap
	angles = list(range(0,15*4+1,15))
	# angles = [0]
	# how many cols to give each angle? limit one, so for large widths the larger angles may not get used
	while True:
		cols_possible = width_cols/cellsize
		cols_each = int(cols_possible/len(angles))
		cols_extra = cols_possible%len(angles)
		# if can't do at least one per angle, drop odd angles
		if not cols_each and math.ceil(cols_extra) < len(angles):
			angles = [angles[i] for i in range(len(angles)) if not (i&0x1)]
		else: break
		
	print("cols per angle:", cols_each, cols_extra)
	offset = Vector()
	prev_negative = False
	cols_total = 0
	cols_done = False
	angle_reverse = False
	while True:
		# oscillate between increasing/decreasing angles until width is met
		for angle in (reversed(angles) if angle_reverse else angles):
			if zigzag and prev_negative:
				angle = -angle
			cols = cols_each
			if angle_reverse:
				cols_extra += 1
			if cols_extra > 0:
				cols += 1
			if not angle_reverse:
				cols_extra -= 1
			for ci in range(cols):
				nxt_corner = skewed_square(bm, rad, angle, offset)
				# fill grid below
				maxy = max(offset.y+cellsize, offset.y+nxt_corner.y)
				oy = -goffset
				while maxy+oy > 0:
					skewed_square(bm, rad, angle, offset+Vector((0, oy, 0)))
					oy -= goffset
				# fill grid above
				miny = min(offset.y, offset.y+nxt_corner.y-cellsize)
				oy = goffset
				while miny+oy < height:
					skewed_square(bm, rad, angle, offset+Vector((0, oy, 0)))
					oy += goffset
				# done?
				cols_total += 1
				cols_done = cols_total*cellsize >= width
				if cols_done: break
				# next offset
				offset += nxt_corner
				offset.y = (offset.y % goffset) + .5*gap
				if zigzag:
					angle = -angle
			if cols_done: break
			prev_negative = angle < 0
		if cols_done: break
		# exceeded width_cols, reverse angles and keep going
		angle_reverse = True

	# trim edges
	H.clip_square(bm, width, height)
	# discard floating edges
	for v in bm.verts:
		if not v.link_faces:
			bm.verts.remove(v)
	
	# extrude
	ret = bmesh.ops.extrude_face_region(bm, geom=H.fltr(bm,True,True,True), use_keep_orig=True)
	bmesh.ops.translate(bm, vec=(0,0,-depth), verts=H.fltr(ret["geom"],True))
	del ret
	# extrude has weird face normals for some reason
	bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

	return bm

def fold_checkerboard(bm, cut, cellsize, depth, angle=45, test=False, *, fltr_co=None, fltr_no=None,):
	""" Create a 45 degree x-axis fold in a checkerboard, adjusting geometry to
		ensure squares have enough surface area contact around edge. Assumes the
		checkerboard is oriented vertically
	"""
	fold_edge_dist = depth/math.tan(.5*(180-angle)*math.pi/180.0)
	angle *= math.pi/180.0
	ctheta = math.cos(angle)
	stheta = math.sin(angle)
	def inner_shift(p):
		c = ctheta
		s = stheta
		w = cellsize
		p2 = p*p
		c2 = c*c
		c21 = c2 + 1
		s2s = s*s - 2*s
		c2s = s2s+c21
		s2 = 2*s-2
		p2p = p2 - p
		return w*(p2p*c2s - s + math.sqrt((s2s - 3*c2 + 1)*p2 + (4*c2 + s2)*p - c2 + 1))/(p2*c2s + p*s2 + 1)

	# cut where we want to fold
	co = Vector((0,0,cut))
	ret = H.clip_plane(bm, co, (0,0,1))
	back_cut_edge = list(filter(lambda g: isinstance(g, bmesh.types.BMVert) and g.co.y > EPSILON, ret["geom_cut"]))
	cut_edges = H.fltr(ret["geom_cut"], e=True)
	del ret
	
	# we want to enforce that after folding, checkerboard squares are attached to the rest
	# of the geometry with width^2 surface area still; this adjusts the geometry at the edge for that purpose;
	# first, build index of backfacing edges by column
	cols = []
	if fltr_co is None:
		faces = bm.faces
	else:
		faces = H.connecting_geometry(H.fltr_plane(bm, fltr_co, fltr_no, v=True, threshold=EPSILON), f=True)
	# if test:
	# 	bmesh.ops.delete(bm, geom=faces, context="FACES")
	# 	return
	for f in bm.faces:
		ctr = f.calc_center_bounds()
		# only care about modifying the back face
		if ctr.y < EPSILON or abs(f.normal.y-1) > EPSILON:
			continue
		# column defined by what multiple of col width the edge midpoint lies in
		col = int(ctr.x // cellsize)
		while len(cols) <= col:
			cols.append([])
		# identify top/bottom edges of face (at most two each), discarding sides
		top = []
		bottom = []
		cut_side = 0 # 0 no cut, 1 = top, -1 = bottom
		flat_mask = 0
		for l in f.loops:
			a = l.vert.co
			b = l.link_loop_next.vert.co
			if abs(a.x-b.x) > EPSILON:
				flat = abs(b.z-a.z) < EPSILON
				is_cut = flat and abs(b.z-cut) < EPSILON
				if b.x < a.x:
					a,b = b,a
					if is_cut: cut_side = -1
					if flat: flat_mask |= 0b1
					side = bottom
				else:
					if is_cut: cut_side = 1
					if flat: flat_mask |= 0b10
					side = top
				side.append({
					"edge":l.edge,
					"va":a, # smaller x
					"vb":b, # larger x
					"flat":flat,
					"cut": is_cut,
					"cut_side": 1 if a.z >= cut-EPSILON else -1
				})
		top.sort(key=lambda e: e["va"].x)
		bottom.sort(key=lambda e: e["va"].x)
		cols[col].append({
			"top":top,
			"bottom":bottom,
			"ctr": ctr,
			"cut": cut_side,
			"flat": flat_mask,  # 0b1 = bottom, 0b10 = top
		})

	# now process each column
	#print("CUT --------", cut, fold_edge_dist)
	for ci,c in enumerate(cols):
		c.sort(key=lambda f: f["ctr"].z)
		# merge faces to give us the full negative squares
		neg_squares = []
		merge_next = False
		for r in c:
			b = list(filter(lambda e: not e["cut"], r["bottom"]))
			t = list(filter(lambda e: not e["cut"], r["top"]))
			# connected to previous by cut edge
			if merge_next and r["cut"] < 0:
				last = neg_squares[-1]
				last["bottom"].extend(b)
				last["top"].extend(t)
			else:
				merge_next = r["cut"] > 0
				neg_squares.append({
					"bottom": b,
					"top": t
				})
		# positive squares are the gaps in neg squares
		for si in range(0, len(neg_squares)-1):
			bottom = neg_squares[si]["top"]
			top = neg_squares[si+1]["bottom"]
			last = si == len(neg_squares)-2
			hi = top[-1]["vb"].z
			lo = bottom[0]["va"].z
			# no need to adjust, outside adjustable range
			if hi <= cut-fold_edge_dist or lo >= cut+fold_edge_dist:
				continue
			# intersect case
			if hi > cut and lo < cut:
				# find p closest to 0.5; pstart >= pend
				pstart = max(1,(cut-lo)/cellsize)
				pend = min(0,1-(hi-cut)/cellsize)
				p = pstart if pstart < .5 else (pend if pend > .5 else 0.5)
				# calculate offset
				t = inner_shift(p)
				offsets = [t*p + fold_edge_dist, t*(1-p) + fold_edge_dist]
			else:
				offsets = [fold_edge_dist - (cut-hi if hi <= cut else lo-cut)]*2
			#print(lo, hi, offsets)
			# do adjustment
			for i,side in enumerate([bottom,top]):
				o = offsets[i]
				for e in side:
					dir = e["cut_side"]*o
					verts = list(e["edge"].verts)
					# this is to deal with cases where there are two cuts, and squares spans both;
					# this won't work if more than one square cross the top cut, but that shouldn't be a problem with angles we're using
					if last and i:
						verts.pop()
					for v in verts:
						v.co.z += dir
	# adjust cut edges
	bmesh.ops.translate(bm, verts=back_cut_edge, vec=(0,0,fold_edge_dist))

	# upper/lower geometry
	fold_geo = list(filter(lambda v: v.co.z > cut+EPSILON, bm.verts))
	# finally perform the fold
	bmesh.ops.rotate(bm,
		verts=fold_geo,
		cent=(0,0,cut),
		matrix=Matrix.Rotation(-angle, 4, "X"))

	# rip at cut, which is needed for later boolean ops for cutting "borders" out
	edges = bmesh.ops.split_edges(bm, edges=cut_edges)["edges"]
	H.fill_edges_with_faces(bm, edges)

def dissolve_ngon(bm):
	""" Dissolves vertices of an ngon that are not on a corner """
	dissolve = []
	for v in bm.verts:
		if len(v.link_edges) <= 2:
			dissolve.append(v)
	bmesh.ops.dissolve_verts(bm, verts=dissolve)

def merge_coplanar_faces(bm):
	""" We needed to rip faces to make rotation/clipping work. Unfortunately boolean op doesn't
		like the coplanar faces and gives artifacts, so this merges the coplanar faces

		This just looks for faces that share coordinates of 3 of its vertices with another face.
		So actually more restricted than "coplanar"
	"""
	# first remove doubles within individual faces;
	# if you don't do this, finding 3 shared vertices criteria won't work
	H.remove_doubles_faces_indiv(bm)

	# find doubles is not a symmetric map
	dbls = bmesh.ops.find_doubles(bm, verts=bm.verts, dist=1e-4)["targetmap"]
	groups = {}
	for d1,d2 in dbls.items():
		# d2 is the vertex that is kept, so is the "owner" of the group
		if d2 in groups:
			group = groups[d2]
		else:
			group = [d2]
			groups[d2] = group
		group.append(d1)
		groups[d1] = group

	# match results
	matched = set()
	keep = set() # vertices to be kept
	avg = defaultdict(list) # vertex to keep -> [vertices to average with]
	weld = {} # vertex to weld -> vertex to keep
	def mark_vertex(a, b, need_avg):
		# secondary merge?
		if a in weld: a = weld[a]
		if b in weld: b = weld[b]
		# already welding
		if a == b: return
		# keep vertex A
		if b in keep:
			b,a = a,b
		# TODO: could handle this if needed
		assert b not in keep
		keep.add(a)
		if need_avg:
			avg[a].append(b)
		weld[b] = a

	def closest_vertex(a, lst):
		""" Find closest point in lst to point a """
		best = None
		best_dist = float("inf")
		for i,b in enumerate(lst):
			dist = (a.co-b.co).length
			if dist < best_dist:
				best = i
				best_dist = dist
		return best, best_dist

	for fa in bm.faces:
		if fa in matched:
			continue
		counts = defaultdict(list) # face B -> shared vertices w/ face A
		for va in fa.verts:
			if va not in groups: continue
			group = groups[va]
			for vb in group:
				if va == vb: continue
				dbl = (va,vb)
				for fb in vb.link_faces:
					assert fa != fb # since we removed doubles from indiv faces before
					counts[fb].append(dbl)
		# must have 3 shared vertices to merge
		for fb,dbls in counts.items():
			if len(dbls) > 2:
				la = len(fa.verts)
				lb = len(fb.verts)
				# if la != lb:
				# 	print("bad")
				# 	continue
				infer = max(la,lb)-len(dbls)
				for a,b in dbls:
					mark_vertex(a,b,False)
				if infer > 0:
					av, bv = zip(*dbls)
					anv = list(filter(lambda v: v not in av, fa.verts))
					bnv = list(filter(lambda v: v not in bv, fb.verts))
					# if unequal number of vertices, we can't merge the faces by merging vertices, we'd
					# need to create extra vertices on one face or delete from another; will use the delete
					# strategy here since it is simpler, merging extra vertices with a neighbor
					if la != lb:
						print("Warning! unequal number of vertices when merging faces:", la, lb)
					# must keep all vertices of smaller face
					if la > lb:
						bnv,anv = anv,bnv
					# greedy, match by closest distance; should work fine for "normal" geometry
					for a in anv:
						best, best_dist = closest_vertex(a, bnv)
						mark_vertex(a, bnv[best], True)
						del bnv[best]
					# leftovers can get merged a second time; again, we'll just greedily match here
					av = (fb if la > lb else fa).verts[:]
					for b in bnv:
						best, best_dist = closest_vertex(b, av)
						mark_vertex(b, av[best], True)
				matched.add(fb)
	
	# perform merge
	for a,others in avg.items():
		for o in others:
			a.co += o.co
		a.co /= (len(others)+1)
	bmesh.ops.weld_verts(bm, targetmap=weld)

def delete_interior_faces(bm):
	""" For this function, an interior face is one where all edges are attached to 3+ faces """
	interior = []
	for f in bm.faces:
		interior_count = 0
		for e in f.edges:
			if len(e.link_faces) > 2:
				interior_count += 1
		if interior_count == len(f.edges):
			interior.append(f)
	bmesh.ops.delete(bm, geom=interior, context="FACES_ONLY")

def remove_internal_geometry(bm, planes, threshold=-EPSILON):
	""" remove geometry that is completely inside and would create an internal hole in the model """
	verts = set()
	for co,no in planes:
		verts.update(H.fltr_plane(bm, Vector(co), H.normed_vector(no), v=True, threshold=threshold))
	verts = H.connecting_geometry(list(verts), True)
	internal = []
	for v in bm.verts:
		if v not in verts:
			internal.append(v)
	bmesh.ops.delete(bm, geom=internal)

def angled_checkerboard(coll, cellsize, shift=0, mirror=False):
	""" Builds checkerboard on 5 different face angles """
	# we will draw pattern at zero degree and 45 degree
	width_true = 4 # set by support_weight script
	outset = .025 # add a small outset to width so boolean op will be stable
	border = .1 # radius, actual border will be 2x this
	depth = min(.4+outset, cellsize*1.107651)

	outset_x = outset*math.sin(67.5*math.pi/180)
	outset_y = outset*math.cos(67.5*math.pi/180)
	width = width_true+outset_x+outset_y
	main_width = width*0.585786437626905 # width = x + x*cos(45)
	corner_width = width-main_width
	corner_skew = corner_width*0.7071067811865476 # corner_width * cos(45)
	corner_height = corner_width*1.220774588761456 # corner_width / cos(35)
	# three separate heights, since the top/bottom can be extended to fill full object
	h1 = main_width + 1.69 + 1.2 - .5
	h2 = main_width # do not changeg
	h3 = main_width + 1.25
	
	print("depth:", depth)
	print("Angled board dims:", main_width, main_width-2*border, corner_width, corner_skew)

	# using the largest angle, since that makes the corners bordering an angle change look cleaner
	b35 = b45 = b55 = border_angle(border, depth, 55)
	# b45 = border_angle(border, depth, 45)
	# b35 = border_angle(border, depth, 35)

	# Back side ----------------------
	side0 = create_checkerboard(cellsize, width-border*1.7071067811865475, h1+h2+h3, depth, main_width-2*border)
	bmesh.ops.rotate(side0, verts=H.fltr(side0,True), matrix=Matrix.Rotation(90*math.pi/180, 4, "X"))
	n1 = Matrix.Rotation((.5*(180-45)-90)*math.pi/180,3,"X") @ Vector((0,0,1))
	fold_checkerboard(side0, h1+h2, cellsize, depth)
	fold_checkerboard(side0, h1, cellsize, depth, test=True, fltr_co=Vector((0,0,h1+h2)), fltr_no=-n1)

	c1 = Vector((main_width-2*border,0,h1))
	n2 = Matrix.Rotation(-45*math.pi/180,3,"X") @ n1
	c2 = Vector((main_width-2*border,corner_width,h1+corner_width))
	c3 = c2+Vector((0,1.1*border,0))
	# bottom
	bn = Matrix.Rotation(b45,3,"Z") @ Vector((1,0,0))
	bottom = H.connecting_geometry(H.fltr_plane(side0, c1, -n1, v=True, threshold=EPSILON), True, True, True)
	H.clip_plane(side0, c1, bn, geo=bottom, delete=True, make_faces=True)
	# angled side
	mn = Matrix.Rotation(b35,3,H.normed_vector((0,1,1))) @ Vector((1,0,0))
	middle = H.connecting_geometry(
		H.fltr_plane(
			H.fltr_plane(side0, c1, n1, v=True, threshold=EPSILON),
			c2, -n2, v=True, threshold=EPSILON
		), True, True, True
	)
	H.clip_plane(side0, c1, mn, geo=middle, delete=True, make_faces=True)
	#top
	top = H.connecting_geometry(H.fltr_plane(side0, c2, n2, v=True, threshold=EPSILON), True, True, True)
	tn1 = Matrix.Rotation(b35,3,"Y") @ Vector((1,0,0))
	tn2 = Matrix.Rotation(b55,3,H.normed_vector((1,1,0))) @ H.normed_vector((1,-1,0))
	H.clip_plane(side0, c2, tn1, geo=top, rip=True, make_faces=True)
	H.clip_plane(side0, c3, tn2, geo=top, rip=True, make_faces=True)
	cutout = H.connecting_geometry(
		H.fltr_plane(
			H.fltr_plane(top, c2, tn1, v=True, threshold=EPSILON),
			c3, tn2, v=True, threshold=EPSILON
		),
		True, True, True
	)
	bmesh.ops.delete(side0, geom=cutout)

	remove_internal_geometry(side0, [
		((0,0,0), (0,-1,0)),
		((0,0,h1), Matrix.Rotation(-45*math.pi/180,3,"X") @ Vector((0,-1,0))),
		((0,0,h1+corner_width), Vector((0,0,1)))
	])

	bmesh.ops.translate(side0, vec=(border,-outset_x,0), verts=H.fltr(side0, True))

	# Angled triangle side ------------------
	side45 = create_checkerboard(cellsize, main_width+corner_skew-2*border, h1+corner_height-border, depth, main_width-2*border)
	bmesh.ops.rotate(side45, verts=H.fltr(side45,True), matrix=Matrix.Rotation(90*math.pi/180, 4, "X"))
	fold_checkerboard(side45, h1, cellsize, depth, 35)

	n1 = Matrix.Rotation((.5*(180-35)-90)*math.pi/180,3,"X") @ Vector((0,0,1))
	c1 = Vector((0,0,h1))
	c2 = (Matrix.Rotation(-35*math.pi/180,3,"X") @ Vector((0,0,corner_height-border))) + Vector((main_width+corner_skew-border,0,h1))
	# bottom
	bn = Matrix.Rotation(-b45,3,"Z") @ Vector((-1,0,0))
	bottom = H.connecting_geometry(H.fltr_plane(side45, c1, -n1, v=True, threshold=EPSILON), True, True, True)
	H.clip_plane(side45, c1, bn, geo=bottom, delete=True, make_faces=True) # left
	H.clip_plane(side45, (main_width-2*border,0,0), (1,0,0), geo=H.fltr_valid(bottom), delete=True, make_faces=True) # right
	# top
	lside_norm = Matrix.Rotation(-35*math.pi/180,3,"X") @ Matrix.Rotation(30*math.pi/180,3,"Y") @ Matrix.Rotation(-b35,3,"Z") @ Vector((-1,0,0))
	rside_norm = Matrix.Rotation(-35*math.pi/180,3,"X") @ Matrix.Rotation(30*math.pi/180,3,"Y") @ Vector((1,0,0))
	top_norm = Matrix.Rotation(-35*math.pi/180 - b55,3,"X") @ Vector((0,0,1))
	def get_top():
		return H.connecting_geometry(H.fltr_plane(side45, c1, n1, v=True, threshold=EPSILON), True, True, True)
	H.clip_plane(side45, c1, lside_norm, geo=get_top(), delete=True, make_faces=True) # left
	H.clip_plane(side45, c2, top_norm, geo=get_top(), delete=True, make_faces=True) # top
	H.clip_plane(side45, (main_width-2*border, 0, h1), rside_norm, geo=get_top(), delete=True, make_faces=True) # right

	remove_internal_geometry(side45, [
		((0,0,0), (0,-1,0)),
		((0,0,h1), Matrix.Rotation(-35*math.pi/180,3,"X") @ Vector((0,-1,0)))
	])

	bmesh.ops.transform(side45, verts=H.fltr(side45, True), matrix=(
		Matrix.Translation((main_width,-outset_x,0)) @ Matrix.Rotation(45*math.pi/180, 4, "Z") @ Matrix.Translation((border,0,0))
	))

	def quick_dissolve(g):
		bmesh.ops.dissolve_limit(g,
			angle_limit=10*math.pi/180,
			use_dissolve_boundaries=True,
			verts=H.fltr(g,v=True),
			edges=H.fltr(g,e=True)
		)
	for g in [side0,side45]:
		quick_dissolve(g)
		merge_coplanar_faces(g)
		delete_interior_faces(g)
		quick_dissolve(g)	

	name = f"checkerboard_{cellsize}"
	mesh = bpy.data.meshes.new(name)
	side45.to_mesh(mesh)
	side45.free()
	side0.from_mesh(mesh)
	if True and (shift or mirror):
		if mirror:
			mat = Matrix.Translation((width_true*(shift+1),0,0)) @ Matrix.Scale(-1,4,(1,0,0))
		else:
			mat = Matrix.Translation((width_true*shift,0,0))
		bmesh.ops.transform(side0, matrix=mat, verts=H.fltr(side0, True))
	bmesh.ops.recalc_face_normals(side0, faces=H.fltr(side0, f=True))
	side0.to_mesh(mesh)
	side0.free()

	obj = bpy.data.objects.new(name, mesh)
	coll.objects.link(obj)
	return obj


coll = bpy.data.collections.new('checkerboard')
bpy.context.scene.collection.children.link(coll)
for i, width in enumerate([.8,.6,.5,.4,.35,.3,.25,.2]):
	# for i, width in enumerate([.35]):
	obj = angled_checkerboard(coll, width, shift=i, mirror=not (i & 0x1))
	
"""
coll = bpy.data.collections.new('checkerboard_flat')
bpy.context.scene.collection.children.link(coll)
for i, width in enumerate([.8,.6,.5,.4,.35,.3,.25,.2]):
	obj = angled_checkerboard(coll, width, shift=i, mirror=not (i & 0x1))


for i in range(3):
	for j in range(3):
		create_checkerboard()
"""