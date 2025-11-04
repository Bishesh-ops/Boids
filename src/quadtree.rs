use ggez::glam::Vec2;

/// A simplified point used by the Quadtree for efficiency.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub position: Vec2,
    pub velocity: Vec2,
}

/// A rectangle defined by its center and half-dimensions (w, h).
#[derive(Debug, Clone, Copy)]
pub struct Rectangle {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rectangle {
    /// Checks if a point is within the rectangle's bounds.
    pub fn contains(&self, point: &Point) -> bool {
        (point.position.x >= self.x - self.w)
            && (point.position.x < self.x + self.w)
            && (point.position.y >= self.y - self.h)
            && (point.position.y < self.y + self.h)
    }

    /// Checks if another rectangle intersects with this one.
    pub fn intersects(&self, range: &Rectangle) -> bool {
        !(range.x - range.w > self.x + self.w
            || range.x + range.w < self.x - self.w
            || range.y - range.h > self.y + self.h
            || range.y + range.h < self.y - self.h)
    }
}

/// The Quadtree structure for spatial partitioning.
pub struct Quadtree {
    pub boundary: Rectangle,
    pub capacity: usize,
    pub points: Vec<Point>,
    pub divided: bool,
    pub northwest: Option<Box<Quadtree>>,
    pub northeast: Option<Box<Quadtree>>,
    pub southwest: Option<Box<Quadtree>>,
    pub southeast: Option<Box<Quadtree>>,
}

impl Quadtree {
    pub fn new(boundary: Rectangle, capacity: usize) -> Self {
        Quadtree {
            boundary,
            capacity,
            points: Vec::new(),
            divided: false,
            northwest: None,
            northeast: None,
            southwest: None,
            southeast: None,
        }
    }

    pub fn subdivide(&mut self) {
        let x = self.boundary.x;
        let y = self.boundary.y;
        let w = self.boundary.w / 2.0;
        let h = self.boundary.h / 2.0;

        let nw = Rectangle {
            x: x - w,
            y: y - h,
            w,
            h,
        };
        self.northwest = Some(Box::new(Quadtree::new(nw, self.capacity)));
        let ne = Rectangle {
            x: x + w,
            y: y - h,
            w,
            h,
        };
        self.northeast = Some(Box::new(Quadtree::new(ne, self.capacity)));
        let sw = Rectangle {
            x: x - w,
            y: y + h,
            w,
            h,
        };
        self.southwest = Some(Box::new(Quadtree::new(sw, self.capacity)));
        let se = Rectangle {
            x: x + w,
            y: y + h,
            w,
            h,
        };
        self.southeast = Some(Box::new(Quadtree::new(se, self.capacity)));

        self.divided = true;
    }

    pub fn insert(&mut self, point: Point) -> bool {
        if !self.boundary.contains(&point) {
            return false;
        }

        if self.points.len() < self.capacity {
            self.points.push(point);
            return true;
        } else {
            if !self.divided {
                self.subdivide();
            }
            if self.northeast.as_mut().unwrap().insert(point) {
                return true;
            }
            if self.northwest.as_mut().unwrap().insert(point) {
                return true;
            }
            if self.southeast.as_mut().unwrap().insert(point) {
                return true;
            }
            if self.southwest.as_mut().unwrap().insert(point) {
                return true;
            }
        }
        false
    }

    /// Finds all points within a given rectangular range.
    pub fn query(&self, range: &Rectangle, found: &mut Vec<Point>) {
        if !self.boundary.intersects(range) {
            return;
        }

        for p in &self.points {
            if range.contains(p) {
                found.push(*p);
            }
        }

        if self.divided {
            self.northwest.as_ref().unwrap().query(range, found);
            self.northeast.as_ref().unwrap().query(range, found);
            self.southwest.as_ref().unwrap().query(range, found);
            self.southeast.as_ref().unwrap().query(range, found);
        }
    }
}

// Basic unit test for the quadtree
#[cfg(test)]
mod tests {
    use super::*;
    use ggez::glam::Vec2;

    #[test]
    fn insert_and_query() {
        let boundary = Rectangle {
            x: 50.0,
            y: 50.0,
            w: 50.0,
            h: 50.0,
        };
        let mut qt = Quadtree::new(boundary, 1);
        let p1 = Point {
            position: Vec2::new(10.0, 10.0),
            velocity: Vec2::ZERO,
        };
        let p2 = Point {
            position: Vec2::new(90.0, 90.0),
            velocity: Vec2::ZERO,
        };
        qt.insert(p1);
        qt.insert(p2);

        let range = Rectangle {
            x: 50.0,
            y: 50.0,
            w: 100.0,
            h: 100.0,
        };
        let mut found = Vec::new();
        qt.query(&range, &mut found);
        assert!(found.len() >= 2);
    }
}
