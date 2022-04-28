import math

def dist(p1,p2):
	print("dist:", math.sqrt((p2[0]-p1[0])**2 + (p2[1] - p1[1])**2), width)

width = .7
depth = min(width+.1, .4)
p = .5 # >= .5
O_s1 = p*width
O_s2 = (1-p)*width


D_s1 = depth
D_s2 = depth

upper_bound_offset = depth*(math.sqrt(2)-1) + width*math.sqrt(1-math.sqrt(2)/2)
print("upper bound offset:", upper_bound_offset)

d = depth
o = width*.5
ap = (25.93396700562964-22.5)/(45-22.5)
t = math.tan((ap*45+(1-ap)*22.5)*math.pi/180.0)
p1 = (d, -d*t)
p2 = ((o+d*(t+1))/math.sqrt(2), (o*(math.sqrt(2)+1) + d*(t-1))/math.sqrt(2))

s2 = math.sqrt(2)
# t = ((s2 - 1)*d*d + d*(-o + s2 + 1) - math.sqrt(d*o*(-d*(-8*o+s2+1) + 7*(3 + 2*s2)*o - 5*s2 - 7)))/(d*(d + 2*s2 + 3))
# print("GOT T:", t)
dist(p1,p2)
a = o+d*(t+1-s2)
b = o*(s2+1)+d*((1+s2)*t-1)
"""
8o^2 = a^2 + b^2
a = t*d + [o + d*(1-s2)]
b = t*d*(1+s2) + [o*(s2+1) - d]
	x = [o + d*(1-s2)]
	y = [o*(s2+1) - d]
	z = 8*o^2
	v = d
	w = d*(1+s2)

z = [tv+x]^2 + [tw+y]^2
z = (tv)^2 + 2*tv*x + x^2 + (tw)^2 + 2*tw*y + y^2
0 = t^2*[v^2 + w^2] + t*[2*v*x + 2*w*y] + [x^2 + y^2 - z]
	a = [v^2 + w^2]
	b = [2*v*x + 2*w*y]
	c = [x^2 + y^2 - z]
t = (-b +- sqrt(b^2 - 4ac))/(2a)

simplified:
	a = d^2*(4+2s2)
	  = 6.82842712474619*d^2
	b = 2*(d*[o + d*(1-s2)] + d*(1+s2)*[o*(s2+1) - d])
	  = 2*(d*o + d^2(1-s2) + d*o*(3+2s2) - d^2*(1+s2)))
	  = 4*d*(o*(2+s2) - d*s2)
	  = (8+4s2)*d*o - 4*s2*d^2
	  = 13.65685424949238*d*o - 5.656854249492381*d^2
	c = [o + d*(1-s2)]^2 + [o*(s2+1) - d]^2 - 8*o^2
	  = o^2 + d^2*(3-2s2) + 2*o*d*(1-s2) + o^2*(3+2s2) + d^2 - 2*d*o*(s2+1) - 8*o^2
	  = o^2*(-4+2s2) + d^2*(4-2s2) - 4*o*d*s2
	  = -1.1715728752538097*o^2 + 1.1715728752538097*d^2 - 5.656854249492381*o*d

For per-column cutoffs, it is a generalization of above:
2*(o1+o2)^2 = a^2 + b^2
a = t*d + [o2 + d*(1-s2)]
b = t*d*(1+s2) + [o1*s2 + o2 - d]
	x = [o2 + d*(1-s2)]
	y = [o1*s2 + o2 - d]
	z = 2*(o1+o2)^2
	v = d
	w = d*(1+s2)
simplified:
	a = same
	b = 2*(d*o2 + d^2(1-s2) + d*o1*(2+s2) + d*o2*(1+s2) - d^2*(1+s2)))
	  = (4+2s2)*d*(o1+o2) - 4*s2*d^2
	  = 6.82842712474619*d*(o1+o2) - 5.656854249492381*d^2
	c = [o2 + d*(1-s2)]^2 + [o1*s2 + o2 - d]^2 - 2*(o1+o2)^2
	  = o2^2 + d^2*(3-2s2) + 2*o2*d*(1-s2) + 2*o1^2 + 2*o1*o2*s2 - 2*o1*s2*d + o2^2 - 2*o2*d + d^2 - 2*(o1+o2)^2
	  = o1*o2*(-4+2*s2) + d^2*(4-2s2) - 2s2*d*(o1+o2)
	  = -1.1715728752538097*o1*o2 + 1.1715728752538097*d^2 - 2.8284271247461903*d*(o1+o2)

New parameterization, with independent angles for top/bottom:
2*w^2 = A^2 + B^2
	A = (1-p)w + d + t*p*w - d*s2
	  = t[p*w] + [w*(1-p) + d*(1-s2)]
	  = tv+x
	B = (1-p)w - d + t*p*w + t*(1-p)*w*s2 + p*w*s2
	  = t[w*(p(1-s2) + s2)] + [w*(p*(s2-1) + 1) - d]
	  = tw+y
simplified:
	a = [p*w]^2 + [w*(p(1-s2) + s2)]^2
	  = w^2*(p^2 + (p(1-s2) + s2)^2)
	  = w^2*2*((2-s2)*(p^2 - p) + 1)
	b = 2*[p*w]*[w*(1-p) + d*(1-s2)] + 2*[w*(p(1-s2) + s2)]*[w*(p*(s2-1) + 1) - d]
	  = w^2*(2s2*p - 2*p^2 + 2) + ((2-2s2)p - 2)*d*w
	c = [w*(1-p) + d*(1-s2)]^2 + [w*(p*(s2-1) + 1) - d]^2 - 2w^2
	  = w^2*((4-2s2)*(p^2 - p)) + d^2*(4-2s2) - 2s2dw
	
	(-b +- sqrt(b2 - 4ac))/2a
	a = ((s2-2)(p^2-p)-1)w
	b = s2d - (2(s2-2)(p^2-p)+s2)w



"""
x = o+d*(1-s2)
y = o*(s2+1) - d
z = 8*o**2
v = d
w = d*(1+s2)
a = v**2 + w**2
b = 2*(v*x + w*y)
c = x**2 + y**2 - z

def optimal_angle(d, o):
	# terms of equation
	dd = d*d
	oo = o*o
	do = d*o
	a = 6.82842712474619*dd
	b = 13.65685424949238*do - 5.656854249492381*dd
	c = -1.1715728752538097*oo + 1.1715728752538097*dd - 5.656854249492381*do
	# quadratic formula for tan(theta)
	tan_angle = (math.sqrt(b**2 - 4*a*c) - b)/(2*a)
	# and return angle
	return math.atan(tan_angle)*180.0/math.pi


angle = optimal_angle(d, o)

dpred = 1/s2*math.sqrt(a**2 + b**2)
print(dpred)
print("optimal:", angle)



cosA = math.cos(45*math.pi/180.0)
sinA = math.sin(45*math.pi/180.0)

p1 = [min(O_s1,D_s1), 0]
p2 = [cosA*O_s2 + sinA*min(O_s2,D_s2), O_s1 + sinA*O_s2 - cosA*min(O_s2,D_s2)]


# dist(p1,p2)