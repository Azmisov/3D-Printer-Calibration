""" Creates labels for the part

	fonts: arial black, dyuthi, verdana bold, open sans, source sans pro, ubuntu
"""
import helpers as H
import bpy, bmesh, math
from mathutils import Matrix

font = bpy.data.fonts.load("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf")

coll = bpy.data.collections.new('labels')
bpy.context.scene.collection.children.link(coll)

for i, width in enumerate([.8,.6,.5,.4,.35,.3,.25,.2]):
	margin = 4.0
	height = 0.8
	depth = 0.4

	# generate text
	name = "."+str(int(width*100))
	txt = bpy.data.curves.new(type="FONT",name=f"txt{name}")
	txt.body = name
	txt.font = font
	txt.resolution_u = txt.render_resolution_u = 2
	lbl = bpy.data.objects.new(name=f"lbl{name}", object_data=txt)

	# convert to mesh
	coll.objects.link(lbl)
	for o in bpy.context.selected_objects:
		o.select_set(False)
	bpy.context.view_layer.objects.active = lbl
	lbl.select_set(True)
	bpy.ops.object.convert(target='MESH')

	# move into position
	bb = H.global_bound_box(lbl)
	size = bb[1]-bb[0]
	ctr = .5*(bb[0]+bb[1])
	g = bmesh.new()
	g.from_mesh(lbl.data)
	# add depth
	extruded = bmesh.ops.extrude_face_region(g, geom=H.fltr(g,True,True,True))["geom"]
	bmesh.ops.translate(g, verts=H.fltr(extruded,True), vec=(0,0,-depth))
	T = Matrix.Translation(-ctr) # center
	T = Matrix.Scale(height/size[1], 4) @ T # scale
	T = Matrix.Rotation(math.pi, 4, "Z") @ T # text facing supports
	T = Matrix.Rotation(-25*math.pi/180, 4, "X") @ T # slight angle to side
	T = Matrix.Translation(((i+.5)*margin, 6, 6.3)) @ T # put into position
	bmesh.ops.transform(g, verts=H.fltr(g,True), matrix=T)
	bmesh.ops.recalc_face_normals(g, faces=H.fltr(g,f=True))

	g.to_mesh(lbl.data)
	g.free()

