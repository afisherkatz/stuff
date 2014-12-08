from operator import itemgetter
import numpy as np
from numpy import linalg
import time
import copy
import cv2

# function to rotate point circularly around origin
def rotate(angle, x, y):
	return (np.cos(angle)*x - np.sin(angle)*y,
			np.sin(angle)*x + np.cos(angle)*y)

def objectRotation(xAngle, yAngle, x, y, z):
	xn = np.cos(-xAngle)*x - np.sin(-xAngle)*z
	zn = np.sin(-xAngle)*x + np.cos(-xAngle)*z
	return (xn,
			np.cos(yAngle)*y - np.sin(yAngle)*zn,
			np.sin(yAngle)*y + np.cos(yAngle)*zn)

def zxyRotate(zAngle, xyAngle, x, y, z):
	xn = np.cos(-zAngle)*x - np.sin(-zAngle)*z
	zn = np.sin(-zAngle)*x + np.cos(-zAngle)*z
	return (np.cos(xyAngle)*xn - np.sin(xyAngle)*y,
			np.sin(xyAngle)*xn + np.cos(xyAngle)*y,
			zn)

def thetaPhiRotate(theta, phi, x, y, z):
	xn = np.cos(phi)*x
	zn = np.sin(phi)*x
	return (xn,
			np.cos(theta)*y - np.sin(theta)*(z - zn),
			np.sin(theta)*y + np.cos(theta)*(z - zn))

def scale(x,y,z):
	return (float(x*150)/float(150-z),
			float(y*150)/float(150-z))
"""
def split(polygon, splitPlane):
	V0 = np.subtract(splitPlane[1], splitPlane[2])
	V0.shape=(3,1)
	V1 = np.subtract(splitPlane[3], splitPlane[1])
	V1.shape=(3,1)
	nPoly = np.roll(polygon, 1)
	lines = [[(x, xn), (y, yn), (z, zn)] for ((x, y, z), (xn, yn, zn)) in zip(polygon, nPoly)]
	for (P1, P2) in zip(polygon,nPoly):
		if ((np.dot(splitPlane[0], np.subtract(splitPlane[1],P1)) >= 0) != (np.dot(splitPlane[0], np.subtract(splitPlane[1], P2) >= 0))):
			Vl = np.subtract(P1,P2)
			Vl.shape=(3,1)
			Vp = np.subtract(P1, splitPlane[1])
			Vp.shape=(3,1)
			m = np.append(Vl, V0, 1)
			m = np.append(m, V1, 1)
			m = linalg.inv(m)
			splitCoord = np.dot(m,Vp)
			splitCoord.shape=(1,3)
			splitCoord = splitCoord[0]



			
			


def plane(polygon):
	if len(polygon) < 3:
		return False
	else:
		(P1, P2, P3) = (*polygon[0:3])
		D1 = [(x1 - x2), (y1 - y2), (z1 - z2) for ((x1, y1, z1), (x2, y2, z2)) in zip(P1, P2)]
		D2 = [(x1 - x2), (y1 - y2), (z1 - z2) for ((x1, y1, z1), (x2, y2, z2)) in zip(P1, P3)]
		V = np.cross(D1, D2)
		return (V, 
				P1,
				P2,
				P3)

class BSP:
	def __init__(self):
		self.list = None
		self.plane = None
		self.BSP_tree = None
	def addPoly()
"""
class Shape:
	def __init__(self):
		self.polygons = []
		self.location = None
		self.phi = None
		self.theta = None

	def addPoly(self, poly, r=0, theta=0, phi=0, zRotation=0, xyRotation=0, rotation=0, xOffset=0, yOffset=0, zOffset=0, color=(255,255,255)):
		points = []

		for coords in poly:
			(x,y) = rotate(rotation, coords[0], coords[1])
			(x,y,z) = thetaPhiRotate(theta, phi, *zxyRotate(zRotation, xyRotation, x, y, r))
			points.append([x+xOffset,y+yOffset,z+zOffset])

		self.polygons.append({'coords':points,
							  'color': color})

	def getShape(self, xRotation=0, yRotation=0, x=150, y=150, z=0, display=np.zeros((300,300,3))):
		polygons = []
		for poly in self.polygons:
			points = []
			for (X,Y,Z) in poly['coords']:
				(X,Y,Z) = objectRotation(xRotation, yRotation, X, Y, Z)
				(pX,pY) = scale(X,Y,Z+z)
				points.append((pX+x,pY+y,Z+z))
			points.insert(0,max(points, key=itemgetter(2))[2])
			points.insert(1,min(points[1:], key=itemgetter(2))[2])
			points.append(poly['color'])
			polygons.append(points)
		polygons = sorted(polygons, key=itemgetter(1))
		polygons = sorted(polygons, key=itemgetter(0))
		polygons = [poly[2:] for poly in polygons]
		for points in polygons:
			color = points.pop(-1)
			points = [p[:-1] for p in points]
			lines = zip(points, np.roll(points,1,axis=0))
			points = np.array(points, dtype = 'int32')
			cv2.fillConvexPoly(display, points, color)
			for ((p1,p2)) in lines:
				p1 = tuple(int(i) for i in p1)
				p2 = tuple(int(i) for i in p2)
				cv2.line(display,p1,p2,color=(0,0,0))
		return copy.deepcopy(display)

	def cube(self, size):
		self.addPoly([[size,size],[size,-size],[-size,-size],[-size,size]],r=size,zRotation=np.pi,color=(255,0,255))
		self.addPoly([[size,size],[size,-size],[-size,-size],[-size,size]],r=size,zRotation=np.pi/2,xyRotation=np.pi,color=(255,255,0))
		self.addPoly([[size,size],[size,-size],[-size,-size],[-size,size]],r=size,zRotation=np.pi/2,xyRotation=-np.pi/2,color=(0,255,255))
		self.addPoly([[size,size],[size,-size],[-size,-size],[-size,size]],r=size,zRotation=np.pi/2,xyRotation=np.pi/2,color=(0,0,255))
		self.addPoly([[size,size],[size,-size],[-size,-size],[-size,size]],r=size,zRotation=np.pi/2,color=(255,0,0))
		self.addPoly([[size,size],[size,-size],[-size,-size],[-size,size]],r=size,color=(0,255,0))
		return self

	def star(self, height, width):
		self.addPoly([[width,height],[0,height],[0,-height],[width,-height]], color=(255,0,0))
		self.addPoly([[width,height],[0,height],[0,-height],[width,-height]], zRotation=np.pi/3, color=(0,255,0))
		self.addPoly([[width,height],[0,height],[0,-height],[width,-height]], zRotation=2*np.pi/3, color=(0,0,255))
		self.addPoly([[width,height],[0,height],[0,-height],[width,-height]], zRotation=np.pi, color=(255,255,0))
		self.addPoly([[width,height],[0,height],[0,-height],[width,-height]], zRotation=4*np.pi/3, color=(255,0,255))
		self.addPoly([[width,height],[0,height],[0,-height],[width,-height]], zRotation=5*np.pi/3, color=(0,255,255))
		return self

	def cog(self):
		self.addPoly([[5,20],[-5,20],[-5,-20],[5,-20]], r=30, color=(255,0,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=np.pi/10, color=(255,255,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=-np.pi/10, color=(0,255,0))
		self.addPoly([[6,20],[-6,20],[-6,-20],[6,-20]], r=20, zRotation=np.pi/6, color=(255,0,0))

		self.addPoly([[5,20],[-5,20],[-5,-20],[5,-20]], r=30, zRotation=np.pi/3 ,color=(255,0,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=np.pi/3+np.pi/10, color=(255,255,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=np.pi/3-np.pi/10, color=(0,255,0))
		self.addPoly([[6,20],[-6,20],[-6,-20],[6,-20]], r=20, zRotation=np.pi/3+np.pi/6, color=(255,0,0))

		self.addPoly([[5,20],[-5,20],[-5,-20],[5,-20]], r=30, zRotation=2*np.pi/3 ,color=(255,0,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=2*np.pi/3+np.pi/10, color=(255,255,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=2*np.pi/3-np.pi/10, color=(0,255,0))
		self.addPoly([[6,20],[-6,20],[-6,-20],[6,-20]], r=20, zRotation=2*np.pi/3+np.pi/6, color=(255,0,0))

		self.addPoly([[5,20],[-5,20],[-5,-20],[5,-20]], r=30, zRotation=np.pi ,color=(255,0,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=np.pi+np.pi/10, color=(255,255,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=np.pi-np.pi/10, color=(0,255,0))
		self.addPoly([[6,20],[-6,20],[-6,-20],[6,-20]], r=20, zRotation=np.pi+np.pi/6, color=(255,0,0))

		self.addPoly([[5,20],[-5,20],[-5,-20],[5,-20]], r=30, zRotation=4*np.pi/3 ,color=(255,0,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=4*np.pi/3+np.pi/10, color=(255,255,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=4*np.pi/3-np.pi/10, color=(0,255,0))
		self.addPoly([[6,20],[-6,20],[-6,-20],[6,-20]], r=20, zRotation=4*np.pi/3+np.pi/6, color=(255,0,0))

		self.addPoly([[5,20],[-5,20],[-5,-20],[5,-20]], r=30, zRotation=5*np.pi/3 ,color=(255,0,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=5*np.pi/3+np.pi/10, color=(255,255,0))
		self.addPoly([[30,20],[20,20],[20,-20],[30,-20]], zRotation=5*np.pi/3-np.pi/10, color=(0,255,0))
		self.addPoly([[6,20],[-6,20],[-6,-20],[6,-20]], r=20, zRotation=5*np.pi/3+np.pi/6, color=(255,0,0))
		return self

	def binary(self, distance, width):
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=distance/2, color=(255,0,0))
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=distance/2 + width, color=(255,255,0))
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=distance/2 + width/2, phi=np.pi/2, xOffset=width/2, color=(0,255,0))
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=distance/2 + width/2, phi=np.pi/2, xOffset=-width/2, color=(0,255,255))
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=distance/2 + width/2, theta=np.pi/2, yOffset=width/2, color=(0,0,255))
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=distance/2 + width/2, theta=np.pi/2, yOffset=-width/2, color=(255,0,255))

		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=-distance/2)
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=-distance/2 - width)
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=-distance/2 - width/2, phi=np.pi/2, xOffset=width/2)
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,]-width/2],[width/2,-width/2]], zOffset=-distance/2 - width/2, phi=np.pi/2, xOffset=-width/2)
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,]]-width/2],[width/2,-width/2]], zOffset=-distance/2 - width/2, theta=np.pi/2, yOffset=width/2)
		self.addPoly([[width/2, width/2],[-width/2, width/2],[-width/2,-width/2],[width/2,-width/2]], zOffset=-distance/2 - width/2, theta=np.pi/2, yOffset=-width/2)
"""
s = Shape()
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=20, color=(255,0,0))]
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=60, color=(0,255,0))

s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=20, zRotation=np.pi/2, color=(255,0,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=60, zRotation=np.pi/2, color=(0,255,0))

s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=20, zRotation=np.pi/2, xyRotation=np.pi/2, color=(255,0,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=60, zRotation=np.pi/2, xyRotation=np.pi/2, color=(0,255,0))

s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=20, zRotation=np.pi/2, xyRotation=-np.pi/2, color=(255,0,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=60, zRotation=np.pi/2, xyRotation=-np.pi/2, color=(0,255,0))

s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=20, zRotation=np.pi/2, xyRotation=np.pi, color=(255,0,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=60, zRotation=np.pi/2, xyRotation=np.pi, color=(0,255,0))

s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=20, zRotation=np.pi, color=(255,0,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=60, zRotation=np.pi, color=(0,255,0))


s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/4, color=(255,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/4, xyRotation=np.pi/2, color=(255,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/4, xyRotation=-np.pi/2, color=(255,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/4, xyRotation=np.pi, color=(255,0,255))

s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=3*np.pi/4, color=(255,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=3*np.pi/4, xyRotation=np.pi/2, color=(255,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=3*np.pi/4, xyRotation=-np.pi/2, color=(255,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=3*np.pi/4, xyRotation=np.pi, color=(255,0,255))

s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/2, xyRotation=np.pi/4, color=(255,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/2, xyRotation=-np.pi/4, color=(255,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/2, xyRotation=3*np.pi/4, color=(255,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/2, xyRotation=-3*np.pi/4, color=(255,0,255))
"""
"""
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=30, zRotation=3*np.pi/4, color=(0,255,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=20, zRotation=np.pi, color=(0,0,255))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi, color=(255,255,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=20, zRotation=np.pi/2, xyRotation=np.pi/2, color=(255,0,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/2, xyRotation=np.pi/2, color=(0,255,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=30, zRotation=np.pi/4, color=(255,255,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=20, zRotation=np.pi/2, xyRotation=-np.pi/2, color=(255,255,0))
s.addPoly([[10,10],[-10,10],[-10,-10],[10,-10]], r=40, zRotation=np.pi/2, xyRotation=-np.pi/2, color=(0,255,255))


xrot = 0
yrot = 0
sTemp = Shape()
s = Shape()
points = []
s.cube(10)
"""

class GUI:
	def __init__(self):
		self.s = Shape()
		self.sTemp = Shape()
		self.a = 0
		self.b = 0
		self.c = 0
		self.x = 0
		self.y = 0
		self.z = 0
		self.xRot = 0
		self.yRot = 0
		cv2.startWindowThread()
		cv2.namedWindow("Shape Editor")
		cv2.createTrackbar("Theta","Shape Editor", 0, 200, self.Theta)
		cv2.createTrackbar("Phi","Shape Editor", 0, 200, self.Phi)
		cv2.createTrackbar("Rotation","Shape Editor", 0, 200, self.R)
		cv2.createTrackbar("X Offset","Shape Editor", 0, 50, self.X)
		cv2.createTrackbar("Y Offset","Shape Editor", 0, 50, self.Y)
		cv2.createTrackbar("Z Offset","Shape Editor", 0, 50, self.Z)
		cv2.createTrackbar("XRot","Shape Editor",0,200,self.XRot)
		cv2.createTrackbar("YRot","Shape Editor",0,200,self.YRot)

	def Theta(self,x):
		self.a = cv2.getTrackbarPos("Theta","Shape Editor")
		return self.a

	def Phi(self,x):
		self.b = cv2.getTrackbarPos("Phi","Shape Editor")
		return self.b

	def R(self,x):
		self.c = cv2.getTrackbarPos("Rotation","Shape Editor")
		return self.c

	def X(self,x):
		self.x = cv2.getTrackbarPos("X Offset","Shape Editor")
		return self.x

	def Y(self,x):
		self.y = cv2.getTrackbarPos("Y Offset","Shape Editor")
		return self.y

	def Z(self,x):
		self.z = cv2.getTrackbarPos("Z Offset","Shape Editor")
		return self.z

	def XRot(self,x):
		self.xRot = cv2.getTrackbarPos("XRot","Shape Editor")
		return self.xRot

	def YRot(self,x):
		self.yRot = cv2.getTrackbarPos("YRot","Shape Editor")
		return self.yRot

	def show(self, s):
		cv2.imshow("Shape Editor",s)

	def run(self):
		while True:
			shape = self.s.getShape(xRotation=self.xRot,yRotation=self.yRot, display=np.zeros((300,300,3)))
			self.show(shape)
			k = cv2.waitKey(0)
			if k == ord('i'):
				print 'type points of convex polygon (type e to exit):'
				points = []
				p = 0
				while p != 'e':
					try:
						point = []
						p = raw_input('point (x): ')
						point.append(int(p))
						p = raw_input('point (y): ')
						point.append(int(p))
						points.append(point)
					except ValueError:
						pass
				self.sTemp.addPoly(points)
				points = self.sTemp.polygons[0]['coords']
				dis = copy.copy(self.s.getShape(xRotation=self.xRot,yRotation=self.yRot, display=np.zeros((300,300,3))))
				shape = self.sTemp.getShape(xRotation=self.xRot,yRotation=self.yRot, display = copy.copy(dis))
				self.show(shape)

				cv2.setTrackbarPos("Theta","Shape Editor",0)
				cv2.setTrackbarPos("Phi","Shape Editor",0)
				cv2.setTrackbarPos("Rotation","Shape Editor",0)
				while True:
					nPoints = []
					z = cv2.waitKey(2)
					if z == ord('e'):
						break
					self.sTemp = Shape()
					self.sTemp.addPoly(points, theta=np.pi*float(self.a)/float(100), phi=np.pi*float(self.b)/float(100), rotation=np.pi*float(self.c)/float(100))
					dis = self.s.getShape(xRotation=np.pi*float(self.xRot)/float(100),yRotation=np.pi*float(self.yRot)/float(100), display=np.zeros((300,300,3)))
					shape = self.sTemp.getShape(xRotation=np.pi*float(self.xRot)/float(100),yRotation=np.pi*float(self.yRot)/float(100), display = copy.copy(dis))
					self.show(shape)
					time.sleep(.1)

				cv2.setTrackbarPos("X Offset","Shape Editor",0)
				cv2.setTrackbarPos("Y Offset","Shape Editor",0)
				cv2.setTrackbarPos("Z Offset","Shape Editor",0)
				while True:
					nPoints = []
					z = cv2.waitKey(2)
					if z == ord('e'):
						break
					self.sTemp = Shape()
					self.sTemp.addPoly(points, xOffset=self.x, yOffset=self.y, zOffset=self.z, rotation=self.c, theta=self.a, phi=self.b)
					dis = self.s.getShape(xRotation=np.pi*float(self.xRot)/float(100),yRotation=np.pi*float(self.yRot)/float(100), display=np.zeros((300,300,3)))
					shape = self.sTemp.getShape(xRotation=np.pi*float(self.xRot)/float(100),yRotation=np.pi*float(self.yRot)/float(100), display=copy.copy(dis))
					self.show(shape)
					time.sleep(.01)	
				self.s.addPoly(points, zOffset=self.z, xOffset=self.x, yOffset=self.y, rotation=self.c, theta=self.a, phi=self.b)
				sTemp = Shape()

g = GUI()
g.run()
#########################################################################
#########################################################################
"""
s = Shape()
cv2.startWindowThread()
cv2.namedWindow("Shape Editor")
cv2.createTrackbar("XY","Shape Editor", 0, 200, retVal)
cv2.createTrackbar("XZ","Shape Editor", 0, 200, retVal)
cv2.createTrackbar("YZ","Shape Editor", 0, 200, retVal)
cv2.createTrackbar("R","Shape Editor", 0, 200, retVal)
while True:

	shape = s.getShape(xRotation=xrot,yRotation=yrot, display=np.zeros((300,300,3)))
	k = cv2.waitKey(0)

	if k == ord('i'):
		print 'type points of convex polygon (type e to exit):'
		points = []
		p = 0
		while p != 'e':
			try:
				point = []
				p = raw_input('point (x): ')
				point.append(int(p))
				p = raw_input('point (y): ')
				point.append(int(p))
				points.append(point)
			except ValueError:
				pass
		sTemp.addPoly(points)
		points = sTemp.polygons[0]['coords']
		dis = copy.copy(s.showShape(xRotation=xrot,yRotation=yrot, display=np.zeros((300,300,3)), ret=1))
		sTemp.showShape(xRotation=xrot,yRotation=yrot, display = copy.copy(dis))
		a = 0
		b = 0
		c = 0
		while True:
			nPoints = []
			z = cv2.waitKey(0)
			a = cv2.getTrackbarPos()
			if z == 65361:
				b += -np.pi/20
			elif z == 65362:
				a += np.pi/20
			elif z == 65363:
				b += np.pi/20
			elif z == 65364:
				a += -np.pi/20
			elif z == 269025062:
				c -= np.pi/20
			elif z == 269025063:
				c += np.pi/20
			elif z == ord('a'):
				xrot -= np.pi/20
			elif z == ord('w'):
				yrot += np.pi/20
			elif z == ord('d'):
				xrot += np.pi/20
			elif z == ord('s'):
				yrot -= np.pi/20
			elif z == ord('e'):
				break
			sTemp = Shape()
			sTemp.addPoly(points, theta=a, phi=b, rotation=c)
			dis = s.showShape(xRotation=xrot,yRotation=yrot, display=np.zeros((300,300,3)), ret=1)
			sTemp.showShape(xRotation=xrot,yRotation=yrot, display = copy.copy(dis))
			time.sleep(.1)

		xOff = 0
		yOff = 0
		zOff = 0
		while True:
			nPoints = []
			z = cv2.waitKey(0)
			if z == 65361:
				xOff -= 1
			elif z == 65362:
				yOff -= 1
			elif z == 65363:
				xOff += 1
			elif z == 65364:
				yOff += 1
			elif z == 269025062:
				zOff -= 1
			elif z == 269025063:
				zOff += 1
			elif z == ord('a'):
				xrot -= np.pi/20
			elif z == ord('w'):
				yrot += np.pi/20
			elif z == ord('d'):
				xrot += np.pi/20
			elif z == ord('s'):
				yrot -= np.pi/20
			elif z == ord('e'):
				break
			sTemp = Shape()
			sTemp.addPoly(points, xOffset=xOff, yOffset=yOff, zOffset=zOff, rotation=c, theta=a, phi=b)
			dis = s.showShape(xRotation=xrot,yRotation=yrot, display=np.zeros((300,300,3)), ret=1)
			sTemp.showShape(xRotation=xrot,yRotation=yrot, display=copy.copy(dis))
			time.sleep(.01)	
		s.addPoly(points, zOffset=zOff, xOffset=xOff, yOffset=yOff, rotation=c, theta=a, phi=b)
		sTemp = Shape()
	elif k == ord('a'):
		xrot -= np.pi/20
	elif k == ord('w'):
		yrot += np.pi/20
	elif k == ord('d'):
		xrot += np.pi/20
	elif k == ord('s'):
		yrot -= np.pi/20



"""
"""
	try:
		s.showShape(xRotation=xrot,yRotation=yrot, display=np.zeros((300,300,3)))
		xrot += .1
		yrot -= .015
		time.sleep(.02)
	except KeyboardInterrupt:
		cv2.destroyWindow("shape")
		cv2.waitKey(1)
		break
"""