"""
Generates a "weight" for a support to hold, with markers indicating volume of resin
"""
import helpers as H
import bpy, bmesh, math
from mathutils import Vector, Matrix
from collections import namedtuple

# vertices
vertices = 64
# angle of support base (degrees)
base_angle = 70
# angle of support tip (degrees)
tip_angle = 85
# base of support height (mm)
base_height = .5
# overhang angle of weight (degrees)
min_angle = 50
# desired radius reduction to create inset volume markers (mm)
inset_delta = .05*3
# min reduced radius as percentage
inset_min_percent = .75
# minimum height of marker (mm);
# I'm using layer_height*desired_layers
marker_min_height = .05*2
# marker volume (mm^3)
marker_volume = (1)**3

FrustrumDims = namedtuple("FrustrumDims", ["tip_rad", "base_rad", "height", "volume", "angle"])
def calculate_frustrum(v, rt):
	""" Computes frustrum that equals a certain volume
		V_cone = (pi*h*r^2)/3
		V_frustrum = pi*h*(r_base^2 + r_base*r_tip + r_tip^2)/3
		
		with preset angle, solve for height
			r_base = r_tip + h/tan(theta)
			h = tan2(theta)
			[solve v = pi*h*((r + h/t)^2 + (r + h/t)*r + r^2)/3 for h]
		with preset height:
			[solve v = pi*h*(rb^2 + rb*rt + rt^2)/3 for rb]
		
		Solve for r_base and h, subject to min_angle and min_height constraints
		
		:param v: target volume
		:param rt: radius at tip of frustrum
	"""
	ta = math.tan(min_angle*math.pi/180.0)
	h = math.pow((rt*ta)**3 + 3/math.pi*v*ta**2, 1/3) - rt*ta   
	if h < marker_min_height:
		h = marker_min_height
		# quadratic solution
		radical = math.sqrt(-3*h*(h*rt**2 - 4*v/math.pi))
		rb = .5*(radical/h - rt)
		#print("min height constraint!", rb)
	else:
		rb = rt + h/ta
	
	a_true = math.atan(h/(rb-rt))*180.0/math.pi
	v_true = math.pi*h*(rb**2 + rb*rt + rt**2)/3
	
	return FrustrumDims(rt, rb, h, v_true, a_true)

def extrude_frustrum(bm, sel, height=0.0, scale=1.0):
	""" Helper to extrude a frustrum from currently selected vertices
		:param bm: bmesh
		:param sel: selection
		:param height: how far to extrude
		:param scale: inset the extruded region by this percentage
	"""
	sel = bmesh.ops.extrude_face_region(bm, geom=sel, use_keep_orig=True)["geom"]
	verts = H.fltr(sel, True)
	transform = Matrix.Identity(4)
	if scale != 1:
		# all points have equal z val
		for v in verts:
			z = v.co.z
			break
		transform = Matrix.Translation((0,0,z)) @ Matrix.Scale(scale, 4) @ Matrix.Translation((0,0,-z)) @ transform
	if height:
		transform = Matrix.Translation((0,0,height)) @ transform
	bmesh.ops.transform(bm, verts=verts, matrix=transform)
	return sel
 
def build_weighted_support(coll, support_diam, offset=0, layers=15):
	# calculate support radii
	# tan(theta) = height/(large_rad - small_rad)
	# large_rad = height/tan(theta) + small_rad
	height_true = 4.5
	width_true = 4
	support_rad = .5*support_diam
	outset = .025 # outset top/bottom for boolean op
	tip_rad = support_diam/math.tan(tip_angle*math.pi/180.0) + support_rad
	base_rad = base_height/math.tan(base_angle*math.pi/180.0) + tip_rad

	bm = bmesh.new()
	bmesh.ops.create_circle(bm, segments=vertices, radius=base_rad, cap_ends=False)
	base_sel = H.fltr(bm, True, True)
	# bottom outset
	sel = extrude_frustrum(bm, base_sel, -outset)
	bmesh.ops.contextual_create(bm, geom=sel)
	# support base and tip
	sel = extrude_frustrum(bm, base_sel, base_height, tip_rad/base_rad)
	sel = extrude_frustrum(bm, sel, support_rad, support_rad/tip_rad)

	# weight attached to support
	total_height = base_height + support_rad
	rad = support_rad
	for i in range(layers):
		dims = calculate_frustrum(marker_volume, rad)
		# print(dims)
		# print("layers:", dims.height/.035, dims.height/.05)
		sel = extrude_frustrum(bm, sel, dims.height, dims.base_rad/rad)
		total_height += dims.height
		# switch to smaller inset at top, for finer lines
		rad_delta = inset_delta if i < 9 else .5*inset_delta
		rad =  max(dims.base_rad - rad_delta, dims.base_rad*inset_min_percent)
		# marker for next weight
		sel = extrude_frustrum(bm, sel, scale=rad/dims.base_rad)
		
	# top outset
	if total_height > height_true:
		raise Exception("true height needs to be larger to accomodate this support!")
	sel = extrude_frustrum(bm, sel, height_true+outset-total_height)
	bmesh.ops.contextual_create(bm, geom=sel)

	# offset
	bmesh.ops.translate(bm, verts=H.fltr(bm,True), vec=((offset+.5)*width_true,0,0))

	# write object
	name = f"support_{support_diam}"
	mesh = bpy.data.meshes.new(name)
	bm.to_mesh(mesh)
	bm.free()
	obj = bpy.data.objects.new(name, mesh)
	coll.objects.link(obj)
			
	print("total height:", total_height)
	print("estimated layers:", total_height/.035, total_height/.05)
	
	return obj


coll = bpy.data.collections.new('supports')
bpy.context.scene.collection.children.link(coll)
diams = [.2,.25,.3,.35,.4,.5,.6,.8]
diams.reverse()
for i,d in enumerate(diams):
	build_weighted_support(coll, d, i)