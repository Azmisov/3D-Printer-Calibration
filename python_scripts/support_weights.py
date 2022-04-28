"""
Generates a "weight" for a support to hold, with markers indicating volume of resin
"""
import bpy, bmesh, math
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

def extrude_frustrum(bm, height, scale):
	""" Helper to extrude a frustrum from currently selected vertices """
	bpy.ops.mesh.extrude_region_move(
		MESH_OT_extrude_region={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False},
		TRANSFORM_OT_translate={
			"value":(0, 0, height),
			"orient_axis_ortho":'X',
			"orient_type":'GLOBAL',
			"orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)),
			"orient_matrix_type":'GLOBAL',
			"constraint_axis":(False, False, True),
			"mirror":False,
			"use_proportional_edit":False,
			"snap":False,
			"gpencil_strokes":False,
			"cursor_transform":False,
			"texture_space":False,
			"view2d_edge_pan":False,
			"release_confirm":False,
			"use_accurate":False,
			"use_automerge_and_split":False
		}
	)
	# regular cylinder? no need to scale
	if scale == 1:
		return
	bpy.ops.transform.resize(
		value=(scale, scale, scale),
		orient_type='GLOBAL',
		orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
		orient_matrix_type='GLOBAL',
		mirror=True,
		use_proportional_edit=False,
		proportional_edit_falloff='SMOOTH',
		proportional_size=1,
		use_proportional_connected=False,
		use_proportional_projected=False
	)
 
def build_weighted_support(coll, support_diam, offset=0, layers=15):
	width_true = 4

	# calculate support radii
	# tan(theta) = height/(large_rad - small_rad)
	# large_rad = height/tan(theta) + small_rad
	support_rad = .5*support_diam
	tip_rad = support_diam/math.tan(tip_angle*math.pi/180.0) + support_rad
	base_rad = base_height/math.tan(base_angle*math.pi/180.0) + tip_rad

	obj = bpy.ops.mesh.primitive_circle_add(
		vertices=vertices,
		radius=base_rad,
		enter_editmode=True,
		align='WORLD',
		location=(offset,0,0),
		scale=(1, 1, 1)
	)
	extrude_frustrum(base_height, tip_rad/base_rad)
	extrude_frustrum(support_rad, support_rad/tip_rad)

	# weight attached to support
	total_height = base_height + support_rad
	rad = support_rad
	for i in range(layers):
		dims = calculate_frustrum(marker_volume, rad)
		print(dims)
		print("layers:", dims.height/.035, dims.height/.05)
		extrude_frustrum(dims.height, dims.base_rad/rad)
		total_height += dims.height
		# switch to smaller inset at top, for finer lines
		rad_delta = inset_delta if i < 9 else .5*inset_delta
		rad =  max(dims.base_rad - rad_delta, dims.base_rad*inset_min_percent)
		# marker for next weight
		extrude_frustrum(0, rad/dims.base_rad)
		
	# to connect with overhang
	extrude_frustrum(4.5-total_height, 1)

	bpy.ops.object.editmode_toggle()
			
	print("total height:", total_height)
	print("estimated layers:", total_height/.035, total_height/.05)
	
	return obj


coll = bpy.data.collections.new('supports')
bpy.context.scene.collection.children.link(coll)
diams = [.2,.25,.3,.35,.4,.5,.6,.8]
diams.reverse()
for i,d in enumerate(diams):
	build_weighted_support(coll, d, i)