//! The global illumination's pure reference layer (M8).
//!
//! The light volume propagation stores the scene's indirect light in a
//! 3D grid of cells: each cell carries one radiance value per cube face
//! (the six dominant directions), injected from the scene's lights and
//! propagated cell to cell over a fixed number of passes. The shading
//! samples the volume at the fragment's world position, choosing the
//! face the fragment's normal faces and interpolating between the
//! neighbouring cells. Every shader in the M8 path translates one of
//! the functions here — the same rule M5 and M7 established: the CPU
//! reference carries the numeric definition, the unit tests pin it,
//! and each backend's readback must match the reference's evaluation
//! on the same inputs.

/// The light volume's grid: cell size, origin and dimensions. Plain
/// data, no graphics-API types — the backends upload it as a 3D
/// texture of `depth` layers.
#[derive(Clone, Debug)]
pub struct LightVolume {
    /// The world-space size of one cell (the grid is uniform).
    pub cell_size: f32,
    /// The world-space origin of the grid's (0, 0, 0) cell's centre.
    pub origin: [f32; 3],
    /// The grid's cell count per axis.
    pub dims: [u32; 3],
    /// The radiance per cell: `dims[0] * dims[1] * dims[2]` entries,
    /// each carrying the six faces' radiance in RGB. The layer index
    /// runs x-major then y then z (the same order a 3D texture's
    /// bytes carry), and the six faces' rgb triplets sit in the
    /// entry's `radiance` array in the FACE order.
    pub cells: Vec<CellRadiance>,
}

/// One cell's radiance: the six cube faces' incoming radiance in RGB.
/// The face order is the FACE_* constants' order.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellRadiance {
    pub faces: [[f32; 3]; 6],
}

/// The face indices the cell arrays carry. The world axes: +X, -X,
/// +Y, -Y, +Z, -Z.
pub const FACE_POS_X: usize = 0;
pub const FACE_NEG_X: usize = 1;
pub const FACE_POS_Y: usize = 2;
pub const FACE_NEG_Y: usize = 3;
pub const FACE_POS_Z: usize = 4;
pub const FACE_NEG_Z: usize = 5;

/// The unit direction vectors the face indices stand for, in the
/// FACE_* order.
pub const FACE_DIRECTIONS: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
];

/// The share of a cell's face radiance that one propagation step
/// sends to the face's neighbouring cell. The 0.5 keeps the
/// propagation stable (the sum of the six shares stays below one) and
/// the value pins the deterministic test's numbers.
pub const PROPAGATION_SHARE: f32 = 0.5;

impl LightVolume {
    /// Creates an empty (all-zero) volume with the given geometry.
    pub fn new(cell_size: f32, origin: [f32; 3], dims: [u32; 3]) -> Self {
        let count = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        LightVolume {
            cell_size,
            origin,
            dims,
            cells: vec![CellRadiance::default(); count],
        }
    }

    /// The number of cells the volume carries.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// The world-space centre of the cell at the given grid index.
    pub fn cell_centre(&self, x: u32, y: u32, z: u32) -> [f32; 3] {
        [
            self.origin[0] + (x as f32 + 0.5) * self.cell_size,
            self.origin[1] + (y as f32 + 0.5) * self.cell_size,
            self.origin[2] + (z as f32 + 0.5) * self.cell_size,
        ]
    }

    /// The grid index of the cell that contains the world position,
    /// clamped into the grid. Out-of-range positions clamp to the
    /// border cells (the volume's sampling never leaves the grid).
    pub fn cell_index(&self, position: [f32; 3]) -> [u32; 3] {
        let mut index = [0u32; 3];
        for axis in 0..3 {
            let offset = position[axis] - self.origin[axis];
            let cell = (offset / self.cell_size).floor().max(0.0) as i64;
            let clamped = cell.min(self.dims[axis] as i64 - 1).max(0);
            index[axis] = clamped as u32;
        }
        index
    }

    /// The flat cell index for the grid coordinates (x-major, then y,
    /// then z — the same order a 3D texture's data carries).
    pub fn flat_index(&self, x: u32, y: u32, z: u32) -> usize {
        let (dx, dy) = (self.dims[0] as usize, self.dims[1] as usize);
        (z as usize) * dx * dy + (y as usize) * dx + (x as usize)
    }

    /// The grid coordinates of a flat cell index (the inverse of
    /// flat_index).
    pub fn grid_index(&self, flat: usize) -> [u32; 3] {
        let (dx, dy) = (self.dims[0] as usize, self.dims[1] as usize);
        let z = flat / (dx * dy);
        let rem = flat % (dx * dy);
        let y = rem / dx;
        let x = rem % dx;
        [x as u32, y as u32, z as u32]
    }

    /// The cell at the grid coordinates (a read convenience).
    pub fn cell(&self, x: u32, y: u32, z: u32) -> &CellRadiance {
        &self.cells[self.flat_index(x, y, z)]
    }

    /// The cell at the grid coordinates (a write convenience).
    pub fn cell_mut(&mut self, x: u32, y: u32, z: u32) -> &mut CellRadiance {
        let flat = self.flat_index(x, y, z);
        &mut self.cells[flat]
    }

    /// Injects a point light's radiance into the volume: the light's
    /// cells (the ones the light's windowed falloff reaches) receive
    /// the intensity at the light's position. The injection matches
    /// the point lights' falloff rule the lighting pass uses: one at
    /// the origin, zero at the radius, the same windowed inverse
    /// square.
    pub fn inject_point_light(&mut self, position: [f32; 3], color: [f32; 3], intensity: f32,
                              radius: f32) {
        if radius <= 0.0 || intensity <= 0.0 {
            return;
        }
        let min = [
            position[0] - radius,
            position[1] - radius,
            position[2] - radius,
        ];
        let max = [
            position[0] + radius,
            position[1] + radius,
            position[2] + radius,
        ];
        let min_index = self.cell_index(min);
        let max_index = self.cell_index(max);
        for z in min_index[2]..=max_index[2] {
            for y in min_index[1]..=max_index[1] {
                for x in min_index[0]..=max_index[0] {
                    let centre = self.cell_centre(x, y, z);
                    let delta = [
                        centre[0] - position[0],
                        centre[1] - position[1],
                        centre[2] - position[2],
                    ];
                    let dist = (delta[0] * delta[0] + delta[1] * delta[1]
                        + delta[2] * delta[2]).sqrt();
                    // The windowed falloff: one at the origin, zero at
                    // the radius (the same rule the point lights use).
                    if dist >= radius {
                        continue;
                    }
                    let falloff = 1.0 - (dist / radius) * (dist / radius);
                    let radiance = falloff * intensity;
                    let flat = self.flat_index(x, y, z);
                    for face in 0..6 {
                        for channel in 0..3 {
                            self.cells[flat].faces[face][channel] +=
                                radiance * color[channel];
                        }
                    }
                }
            }
        }
    }

    /// Injects a directional light (the sun) into the volume: every
    /// cell in the grid receives the intensity (the sun reaches the
    /// whole volume uniformly, the propagation's injection source).
    pub fn inject_directional_light(&mut self, color: [f32; 3], intensity: f32) {
        if intensity <= 0.0 {
            return;
        }
        for cell in self.cells.iter_mut() {
            for face in 0..6 {
                for channel in 0..3 {
                    cell.faces[face][channel] += intensity * color[channel];
                }
            }
        }
    }

    /// Runs one propagation step: every cell sends
    /// PROPAGATION_SHARE of each face's radiance to the neighbour the
    /// face points at (the face's positive axis direction). The send
    /// is a flux: the source cell loses the share it sends and the
    /// neighbour receives it on the same face (the radiance keeps its
    /// travel direction). A face at the grid's border sends outside
    /// the grid (the energy leaves). The step is deterministic and
    /// energy-bounded: the total never grows.
    pub fn propagate_step(&mut self) {
        let dims = self.dims;
        let mut next = self.cells.clone();
        let share = PROPAGATION_SHARE;
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let flat = self.flat_index(x, y, z);
                    // The face-to-neighbour offsets: +X, -X, +Y, -Y,
                    // +Z, -Z (the FACE_* order).
                    let offsets: [(i64, i64, i64); 6] = [
                        (1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (0, 0, 1), (0, 0, -1),
                    ];
                    for face in 0..6 {
                        let (dx, dy, dz) = offsets[face];
                        let nx = x as i64 + dx;
                        let ny = y as i64 + dy;
                        let nz = z as i64 + dz;
                        if nx < 0 || ny < 0 || nz < 0
                            || nx >= dims[0] as i64
                            || ny >= dims[1] as i64
                            || nz >= dims[2] as i64
                        {
                            // The border's flux leaves the grid: the
                            // source still loses the share it would
                            // send (the energy leaves with it).
                            for channel in 0..3 {
                                next[flat].faces[face][channel] -=
                                    share * self.cells[flat].faces[face][channel];
                            }
                            continue;
                        }
                        let neighbour = self.flat_index(nx as u32, ny as u32, nz as u32);
                        for channel in 0..3 {
                            let flux = share * self.cells[flat].faces[face][channel];
                            next[flat].faces[face][channel] -= flux;
                            next[neighbour].faces[face][channel] += flux;
                        }
                    }
                }
            }
        }
        self.cells = next;
    }

    /// Runs the propagation for a fixed number of steps.
    pub fn propagate(&mut self, steps: u32) {
        for _ in 0..steps {
            self.propagate_step();
        }
    }

    /// The face index the surface's normal faces: the face whose
    /// direction has the largest dot product with the normal. The
    /// normal must be non-zero; the caller guarantees it.
    pub fn face_for_normal(normal: [f32; 3]) -> usize {
        let mut best = 0;
        let mut best_dot = f32::NEG_INFINITY;
        for face in 0..6 {
            let dot = normal[0] * FACE_DIRECTIONS[face][0]
                + normal[1] * FACE_DIRECTIONS[face][1]
                + normal[2] * FACE_DIRECTIONS[face][2];
            if dot > best_dot {
                best_dot = dot;
                best = face;
            }
        }
        best
    }

    /// Samples the volume at the world position for the surface whose
    /// normal the given direction faces: the face's radiance,
    /// trilinearly interpolated between the 8 surrounding cells. The
    /// position clamps into the grid: a position outside the grid's
    /// range returns the border cell's radiance (the volume's
    /// sampling never leaves it). The returned radiance is the face's
    /// RGB at the interpolated weight.
    pub fn sample(&self, position: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
        let face = Self::face_for_normal(normal);
        // The position's continuous cell coordinate (the cell centre
        // offset maps to the cell index minus 0.5).
        let mut base = [0u32; 3];
        let mut frac = [0.0f32; 3];
        for axis in 0..3 {
            let offset = position[axis] - self.origin[axis];
            let continuous = offset / self.cell_size - 0.5;
            if continuous <= 0.0 {
                // Below the first cell's centre: the border cell
                // exactly (no neighbour to interpolate toward).
                base[axis] = 0;
                frac[axis] = 0.0;
            } else if continuous >= self.dims[axis] as f32 - 1.0 {
                // Past the last cell's centre: the border cell
                // exactly (the sampling clamps into the grid).
                base[axis] = self.dims[axis] - 1;
                frac[axis] = 0.0;
            } else {
                let floor = continuous.floor();
                base[axis] = floor as u32;
                frac[axis] = continuous - floor;
            }
        }
        let mut result = [0.0f32; 3];
        for corner in 0..8 {
            let cx = base[0] + if corner & 1 != 0 { 1 } else { 0 };
            let cy = base[1] + if corner & 2 != 0 { 1 } else { 0 };
            let cz = base[2] + if corner & 4 != 0 { 1 } else { 0 };
            // The corner's trilinear weight; the axes past the grid
            // clamp to the border cell (the weight collapses).
            let wx = if cx >= self.dims[0] { 0.0 } else if corner & 1 != 0 { frac[0] } else { 1.0 - frac[0] };
            let wy = if cy >= self.dims[1] { 0.0 } else if corner & 2 != 0 { frac[1] } else { 1.0 - frac[1] };
            let wz = if cz >= self.dims[2] { 0.0 } else if corner & 4 != 0 { frac[2] } else { 1.0 - frac[2] };
            let weight = wx * wy * wz;
            if weight == 0.0 {
                continue;
            }
            let clamped_x = cx.min(self.dims[0] - 1);
            let clamped_y = cy.min(self.dims[1] - 1);
            let clamped_z = cz.min(self.dims[2] - 1);
            let flat = self.flat_index(clamped_x, clamped_y, clamped_z);
            for channel in 0..3 {
                result[channel] += weight * self.cells[flat].faces[face][channel];
            }
        }
        result
    }

    /// The volume's total radiance across all cells and faces (the
    /// energy conservation test's measure).
    pub fn total_radiance(&self) -> f32 {
        let mut total = 0.0;
        for cell in self.cells.iter() {
            for face in 0..6 {
                for channel in 0..3 {
                    total += cell.faces[face][channel];
                }
            }
        }
        total
    }

    /// The cell's diffuse irradiance: the six faces' radiance's mean
    /// (the incident irradiance a lambertian surface receives from the
    /// cell's radiance field, the radiance cubed into six directions
    /// the reference's approximation). The shading multiplies the
    /// albedo by this value — the same `albedo * irradiance` rule the
    /// direct lighting's diffuse carries.
    pub fn cell_irradiance(&self, x: u32, y: u32, z: u32) -> [f32; 3] {
        let cell = self.cell(x, y, z);
        let mut mean = [0.0f32; 3];
        for face in 0..6 {
            for channel in 0..3 {
                mean[channel] += cell.faces[face][channel];
            }
        }
        for channel in 0..3 {
            mean[channel] /= 6.0;
        }
        mean
    }

    /// The volume's irradiance sampled at the world position: the
    /// cells' irradiances trilinearly interpolated (the same rule the
    /// face sampling's interpolation follows, applied to the mean).
    pub fn sample_irradiance(&self, position: [f32; 3]) -> [f32; 3] {
        // The position's continuous cell coordinate (the same clamp the
        // face sampling uses).
        let mut base = [0u32; 3];
        let mut frac = [0.0f32; 3];
        for axis in 0..3 {
            let offset = position[axis] - self.origin[axis];
            let continuous = offset / self.cell_size - 0.5;
            if continuous <= 0.0 {
                base[axis] = 0;
                frac[axis] = 0.0;
            } else if continuous >= self.dims[axis] as f32 - 1.0 {
                base[axis] = self.dims[axis] - 1;
                frac[axis] = 0.0;
            } else {
                let floor = continuous.floor();
                base[axis] = floor as u32;
                frac[axis] = continuous - floor;
            }
        }
        let mut result = [0.0f32; 3];
        for corner in 0..8 {
            let cx = base[0] + if corner & 1 != 0 { 1 } else { 0 };
            let cy = base[1] + if corner & 2 != 0 { 1 } else { 0 };
            let cz = base[2] + if corner & 4 != 0 { 1 } else { 0 };
            let wx = if cx >= self.dims[0] { 0.0 } else if corner & 1 != 0 { frac[0] } else { 1.0 - frac[0] };
            let wy = if cy >= self.dims[1] { 0.0 } else if corner & 2 != 0 { frac[1] } else { 1.0 - frac[1] };
            let wz = if cz >= self.dims[2] { 0.0 } else if corner & 4 != 0 { frac[2] } else { 1.0 - frac[2] };
            let weight = wx * wy * wz;
            if weight == 0.0 {
                continue;
            }
            let clamped_x = cx.min(self.dims[0] - 1);
            let clamped_y = cy.min(self.dims[1] - 1);
            let clamped_z = cz.min(self.dims[2] - 1);
            let irradiance = self.cell_irradiance(clamped_x, clamped_y, clamped_z);
            for channel in 0..3 {
                result[channel] += weight * irradiance[channel];
            }
        }
        result
    }
}

/// Converts one f32 to its IEEE 754 binary16 (half float) bits, the
/// same round-to-nearest-even rule the GPU's Rgba16F format carries.
/// The path mirrors the C++ reference's conversion: handle the zero,
/// the subnormal and the infinity cases, then round the 24-bit
/// mantissa to 10 bits.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exponent = ((bits >> 23) & 0xFF) as i32;
    let mut mantissa = bits & 0x7F_FF_FF;
    if exponent == 255 {
        // Infinity or NaN: the 16-bit form carries the same class.
        return sign | 0x7C00 | (if mantissa != 0 { 0x0200 } else { 0 });
    }
    if exponent == 0 && mantissa == 0 {
        return sign;
    }
    if exponent == 0 {
        // Subnormal f32: the 16-bit form's smallest subnormal is
        // larger than the value if the mantissa's top bits vanish;
        // the shift below handles it through the same round path.
        exponent = 1;
        while mantissa & 0x40_00_00 == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= 0x7F_FF_FF;
    }
    // Rebase the exponent onto the 16-bit bias 15.
    let unbiased = exponent - 127;
    let shifted = unbiased + 15;
    if shifted >= 31 {
        // Overflow: the 16-bit form's infinity (the values the
        // engine's radiance carries stay far below this).
        return sign | 0x7C00;
    }
    if shifted <= 0 {
        // Subnormal h: the mantissa shifts right with the round to
        // nearest even.
        let mantissa_full = 0x40_00_00 | mantissa;
        let shift = 14 - shifted;
        let round_bit = 1u32 << (shift - 1);
        let mut result = mantissa_full >> shift;
        if mantissa_full & round_bit != 0
            && (mantissa_full & (round_bit - 1) != 0
                || result & 1 != 0)
        {
            result += 1;
        }
        return sign | result as u16;
    }
    let mut result = ((shifted as u32) << 10) | (mantissa >> 13);
    // Round to nearest even: the 13 dropped bits' guard.
    let round_bit = mantissa & 0x1_00_0;
    let sticky = mantissa & 0x0_FF_F;
    if round_bit != 0 && (sticky != 0 || result & 1 != 0) {
        result += 1;
        if result & 0x7C00 == 0x7C00 {
            // The mantissa's carry overflowed into the exponent: the
            // 16-bit form's next larger class (the infinity at the
            // largest normal's rounding).
            return sign | 0x7C00;
        }
    }
    sign | result as u16
}

/// Converts one binary16 bit pattern back to f32 (the readback's
/// checks' exact round trip).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal h: normalise into the f32 form.
        let mut exponent = 127 - 15 + 1;
        let mut mantissa = mantissa;
        while mantissa & 0x400 == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= 0x3FF;
        return f32::from_bits(
            sign | (exponent << 23) | (mantissa << 13),
        );
    }
    if exponent == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13));
    }
    f32::from_bits(
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    )
}

impl LightVolume {
    /// Packs the volume's cells into the 3D texture's byte stream the
    /// backends' uploads carry: the width `dims[0]`, the height
    /// `dims[1]`, the depth `dims[2] * 6` (the face `f` of the cell
    /// `(x, y, z)` sits at the depth layer `z * 6 + f`), the `Rgba16F`
    /// voxels carrying the radiance in RGB and the alpha 1.0. The
    /// stream's byte order is the texture's row-major x, then y, then
    /// layer order (the same order the fill_texture_3d calls carry).
    pub fn pack_texture_3d(&self) -> Vec<u8> {
        let (width, height, depth) =
            (self.dims[0], self.dims[1], self.dims[2] * 6);
        let mut out = vec![0u8; width as usize
            * height as usize
            * depth as usize
            * 8];
        for z in 0..self.dims[2] {
            for face in 0..6u32 {
                for y in 0..height {
                    for x in 0..width {
                        let cell = self.cell(x, y, z);
                        let radiance = cell.faces[face as usize];
                        let voxel = ((layer_index(z, face) as usize
                            * height as usize
                            + y as usize)
                            * width as usize
                            + x as usize)
                            * 8;
                        for channel in 0..3 {
                            let half =
                                f32_to_f16_bits(radiance[channel]);
                            out[voxel + channel * 2] = (half & 0xFF) as u8;
                            out[voxel + channel * 2 + 1] = (half >> 8) as u8;
                        }
                        // The alpha 1.0's binary16 (0x3C00).
                        out[voxel + 6] = 0x00;
                        out[voxel + 7] = 0x3C;
                    }
                }
            }
        }
        out
    }
}

/// The layer index the face `f` of the cell `(x, y, z)` sits at (the
/// same rule the backends' shaders' layer selection follows).
pub fn layer_index(z: u32, face: u32) -> u32 {
    z * 6 + face
}

#[cfg(test)]
mod pack_tests {
    use super::*;

    #[test]
    fn f32_to_f16_round_trips_through_the_common_values() {
        for value in [
            0.0f32, 1.0, 0.5, 0.35294118, 2.0, 4.0, 100.0, 0.25,
        ]
        .iter()
        {
            let bits = f32_to_f16_bits(*value);
            let back = f16_bits_to_f32(bits);
            // The 16-bit form's precision: the round trip holds the
            // value to the 10-bit mantissa's relative error.
            let error = ((back - value) / value.max(1e-6)).abs();
            assert!(
                error < 0.001,
                "value {} round-tripped to {}",
                value,
                back
            );
        }
    }

    #[test]
    fn pack_texture_3d_carries_the_cells_faces_into_the_layers() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [2, 1, 1]);
        volume.cell_mut(0, 0, 0).faces[FACE_POS_X] = [1.0, 0.5, 0.0];
        volume.cell_mut(1, 0, 0).faces[FACE_NEG_Z] = [0.0, 0.0, 0.25];
        let packed = volume.pack_texture_3d();
        // The texture's shape: 2 wide, 1 tall, 6 deep, 8 bytes per voxel.
        assert_eq!(packed.len(), 2 * 1 * 6 * 8);
        // The face POS_X of the cell (0, 0, 0) sits at the layer 0's
        // voxel (0, 0): the radiance 1.0, 0.5, 0.0 in RGB and the
        // alpha 1.0.
        let voxel = &packed[0..8];
        let r = f16_bits_to_f32(u16::from_le_bytes([voxel[0], voxel[1]]));
        let g = f16_bits_to_f32(u16::from_le_bytes([voxel[2], voxel[3]]));
        let b = f16_bits_to_f32(u16::from_le_bytes([voxel[4], voxel[5]]));
        let a = f16_bits_to_f32(u16::from_le_bytes([voxel[6], voxel[7]]));
        assert_eq!(r, 1.0);
        assert_eq!(g, 0.5);
        assert_eq!(b, 0.0);
        assert_eq!(a, 1.0);
        // The face NEG_Z of the cell (1, 0, 0) sits at the layer 5's
        // voxel (1, 0) (the layer `z * 6 + f` rule).
        let voxel = &packed[(5 * 1 * 2 + 1) * 8..(5 * 1 * 2 + 1) * 8 + 8];
        let b = f16_bits_to_f32(u16::from_le_bytes([voxel[4], voxel[5]]));
        assert_eq!(b, 0.25);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn new_volume_is_all_zero() {
        let volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [4, 4, 4]);
        assert_eq!(volume.cell_count(), 64);
        assert_eq!(volume.total_radiance(), 0.0);
        let sample = volume.sample([1.5, 1.5, 1.5], [0.0, 1.0, 0.0]);
        for channel in 0..3 {
            assert_eq!(sample[channel], 0.0);
        }
    }

    #[test]
    fn cell_index_maps_positions_into_the_grid() {
        let volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [4, 4, 4]);
        // The centre of cell (1, 1, 1) maps back to (1, 1, 1).
        assert_eq!(volume.cell_index([1.5, 1.5, 1.5]), [1, 1, 1]);
        // The origin's cell is (0, 0, 0).
        assert_eq!(volume.cell_index([0.0, 0.0, 0.0]), [0, 0, 0]);
        // Out-of-range positions clamp to the border cells.
        assert_eq!(volume.cell_index([-10.0, -10.0, -10.0]), [0, 0, 0]);
        assert_eq!(volume.cell_index([10.0, 10.0, 10.0]), [3, 3, 3]);
    }

    #[test]
    fn flat_index_round_trips_through_grid_index() {
        let volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [4, 3, 2]);
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    let flat = volume.flat_index(x, y, z);
                    assert_eq!(volume.grid_index(flat), [x, y, z]);
                }
            }
        }
    }

    #[test]
    fn point_light_injection_follows_the_windowed_falloff() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [4, 4, 4]);
        // A point light at the centre of cell (1, 1, 1) with radius 2:
        // the origin cell's falloff is one (distance zero), the
        // neighbours' falloff is 1 - 0.25 = 0.75.
        volume.inject_point_light([1.5, 1.5, 1.5], [1.0, 1.0, 1.0], 1.0, 2.0);
        let centre = volume.cells[volume.flat_index(1, 1, 1)];
        assert!(close(centre.faces[FACE_POS_X][0], 1.0),
            "the origin cell's falloff: {}", centre.faces[FACE_POS_X][0]);
        let neighbour = volume.cells[volume.flat_index(2, 1, 1)];
        assert!(close(neighbour.faces[FACE_POS_X][0], 0.75),
            "the neighbour's falloff: {}", neighbour.faces[FACE_POS_X][0]);
        // The far cell (distance 2.0) receives nothing.
        let far = volume.cells[volume.flat_index(3, 1, 1)];
        assert_eq!(far.faces[FACE_POS_X][0], 0.0);
    }

    #[test]
    fn directional_light_injection_fills_the_volume() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [2, 2, 2]);
        volume.inject_directional_light([1.0, 0.5, 0.25], 0.5);
        for cell in volume.cells.iter() {
            for face in 0..6 {
                assert!(close(cell.faces[face][0], 0.5));
                assert!(close(cell.faces[face][1], 0.25));
                assert!(close(cell.faces[face][2], 0.125));
            }
        }
    }

    #[test]
    fn propagation_moves_half_the_face_radiance_to_the_neighbour() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [4, 1, 1]);
        // Seed cell (1, 0, 0)'s +X face with 1.0 and propagate once.
        volume.cell_mut(1, 0, 0).faces[FACE_POS_X] = [1.0, 0.0, 0.0];
        volume.propagate_step();
        // The neighbour (2, 0, 0)'s +X face received the share.
        assert!(close(volume.cell(2, 0, 0).faces[FACE_POS_X][0], 0.5));
        // The source cell lost the share it sent (the flux leaves).
        assert!(close(volume.cell(1, 0, 0).faces[FACE_POS_X][0], 0.5));
        // The border cell (3, 0, 0)'s +X face sends outside (nothing
        // to receive): the propagation's energy leaves the grid.
        assert!(close(volume.cell(3, 0, 0).faces[FACE_POS_X][0], 0.0));
    }

    #[test]
    fn propagation_three_steps_reaches_three_cells_away() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [8, 1, 1]);
        volume.cell_mut(0, 0, 0).faces[FACE_POS_X] = [1.0, 0.0, 0.0];
        volume.propagate(3);
        // Each step sends half the radiance one cell forward and the
        // source keeps half. After 3 steps the wave reads (the
        // CPU reference's exact values the readback pins):
        // cell 0 = 0.125, cell 1 = 0.375, cell 2 = 0.375,
        // cell 3 = 0.125.
        assert!(close(volume.cell(0, 0, 0).faces[FACE_POS_X][0], 0.125));
        assert!(close(volume.cell(1, 0, 0).faces[FACE_POS_X][0], 0.375));
        assert!(close(volume.cell(2, 0, 0).faces[FACE_POS_X][0], 0.375));
        assert!(close(volume.cell(3, 0, 0).faces[FACE_POS_X][0], 0.125));
        assert!(close(volume.cell(4, 0, 0).faces[FACE_POS_X][0], 0.0));
    }

    #[test]
    fn propagation_never_grows_the_total_energy() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [4, 4, 4]);
        volume.inject_directional_light([1.0, 1.0, 1.0], 1.0);
        let before = volume.total_radiance();
        volume.propagate(5);
        let after = volume.total_radiance();
        assert!(after <= before + 1e-4, "the total grew: {before} -> {after}");
        // The interior's energy survives the leaks (the flux moves it,
        // the borders shed it gradually).
        assert!(after > 0.0, "the interior's energy must survive: {after}");
    }

    #[test]
    fn face_for_normal_picks_the_dominant_direction() {
        assert_eq!(LightVolume::face_for_normal([1.0, 0.0, 0.0]), FACE_POS_X);
        assert_eq!(LightVolume::face_for_normal([-1.0, 0.0, 0.0]), FACE_NEG_X);
        assert_eq!(LightVolume::face_for_normal([0.0, 1.0, 0.0]), FACE_POS_Y);
        assert_eq!(LightVolume::face_for_normal([0.0, -1.0, 0.0]), FACE_NEG_Y);
        assert_eq!(LightVolume::face_for_normal([0.0, 0.0, 1.0]), FACE_POS_Z);
        assert_eq!(LightVolume::face_for_normal([0.0, 0.0, -1.0]), FACE_NEG_Z);
        // A diagonal normal picks the first face that ties (the
        // deterministic rule the readback pins).
        assert_eq!(LightVolume::face_for_normal([0.7, 0.7, 0.0]), FACE_POS_X);
    }

    #[test]
    fn sample_at_a_cell_centre_returns_the_cell_radiance() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [4, 4, 4]);
        volume.cell_mut(2, 1, 1).faces[FACE_POS_X] = [0.7, 0.3, 0.1];
        // The cell centre's sample returns the cell's radiance exactly
        // (the fraction sits at zero).
        let sample = volume.sample([2.5, 1.5, 1.5], [1.0, 0.0, 0.0]);
        assert!(close(sample[0], 0.7));
        assert!(close(sample[1], 0.3));
        assert!(close(sample[2], 0.1));
    }

    #[test]
    fn sample_interpolates_between_the_cells_trilinearly() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [4, 1, 1]);
        volume.cell_mut(1, 0, 0).faces[FACE_POS_X] = [1.0, 0.0, 0.0];
        volume.cell_mut(2, 0, 0).faces[FACE_POS_X] = [0.0, 1.0, 0.0];
        // Halfway between the cells' centres, the sample averages.
        let sample = volume.sample([2.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        assert!(close(sample[0], 0.5));
        assert!(close(sample[1], 0.5));
        // A quarter of the way toward cell 1: the weights follow.
        let sample = volume.sample([1.75, 0.5, 0.5], [1.0, 0.0, 0.0]);
        assert!(close(sample[0], 0.75));
        assert!(close(sample[1], 0.25));
    }

    #[test]
    fn sample_clamps_positions_outside_the_grid() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [2, 2, 2]);
        volume.inject_directional_light([1.0, 1.0, 1.0], 1.0);
        // A position far outside the grid clamps to the border cell
        // exactly (the border cell's radiance, no blend with a
        // neighbour outside the grid).
        let sample = volume.sample([100.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        assert!(close(sample[0], 1.0));
        let sample = volume.sample([-100.0, 0.5, 0.5], [0.0, 1.0, 0.0]);
        assert!(close(sample[1], 1.0));
    }

    #[test]
    fn cell_irradiance_is_the_six_faces_mean() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [2, 2, 2]);
        let mut cell = CellRadiance::default();
        for face in 0..6 {
            cell.faces[face] = [1.0, 0.0, 0.0];
        }
        *volume.cell_mut(0, 0, 0) = cell;
        let irradiance = volume.cell_irradiance(0, 0, 0);
        assert!(close(irradiance[0], 1.0));
        assert!(close(irradiance[1], 0.0));
        // A second cell carries two lit faces: the mean divides by six.
        let mut cell = CellRadiance::default();
        cell.faces[FACE_POS_X] = [3.0, 0.0, 0.0];
        cell.faces[FACE_NEG_X] = [3.0, 0.0, 0.0];
        *volume.cell_mut(1, 0, 0) = cell;
        let irradiance = volume.cell_irradiance(1, 0, 0);
        assert!(close(irradiance[0], 1.0));
    }

    #[test]
    fn sample_irradiance_interpolates_the_cells_mean() {
        let mut volume = LightVolume::new(1.0, [0.0, 0.0, 0.0], [4, 1, 1]);
        let mut cell = CellRadiance::default();
        for face in 0..6 {
            cell.faces[face] = [1.0, 0.0, 0.0];
        }
        *volume.cell_mut(1, 0, 0) = cell;
        // A point light's injection reaches the cell; the irradiance
        // carries the mean.
        let mut cell = CellRadiance::default();
        for face in 0..6 {
            cell.faces[face] = [0.0, 2.0, 0.0];
        }
        *volume.cell_mut(2, 0, 0) = cell;
        // Halfway between the cells' centres, the mean blends.
        let irradiance = volume.sample_irradiance([2.0, 0.5, 0.5]);
        assert!(close(irradiance[0], 0.5));
        assert!(close(irradiance[1], 1.0));
    }
}
