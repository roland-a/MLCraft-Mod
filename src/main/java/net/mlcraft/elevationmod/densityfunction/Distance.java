package net.mlcraft.elevationmod.densityfunction;

import com.mojang.datafixers.util.Either;
import com.mojang.serialization.Codec;
import com.mojang.serialization.MapCodec;
import com.mojang.serialization.codecs.RecordCodecBuilder;

import net.minecraft.util.dynamic.CodecHolder;
import net.minecraft.world.gen.densityfunction.DensityFunction;

/**
 * Density function calculating Euclidean distance from the origin (X=0, Z=0) and returning a value of
 * 0 when the maximum value is reached or exceeded, and 1 at the origin itself. The values inbetween
 * are linearly interpolated. The Y-value (height) is ignored.
 * <br><br>
 * <strong>ID:</strong><code>"mlcraft:distance"</code>
 * <br><br>
 * Parameter:
 * <dl>
 * <dt>"max"</dt><dd>Data type: double<br>
 *   Value range: 1.0 to 1000000000.0<br>
 *   The distance at and beyond which this function returns 0.0</dd>
 * <dt>"min"</dt><dd>Data type: double</br>
 *   Value range: 0.0 to 1000000000.0<br>
 *   Optional; default value 0.0<br>
 *   The distance at and before which this function returns 1.0</dd>
 * <dt>Return value</dt><dd>1.0 exactly at the origin (X=0, Z=0) and up to "min" distance from it.<br>
 *   0.0 exactly at the distance specified in "max" and beyond it.<br>
 *   Linearly in respect to distance interpolated value between those two.</dd>
 * </dl>
 * <br><br>
 * Example:
 * <pre>
 * {
 *  "type": "mlcraft:distance",
 *  "max": 1000
 * }
 * </pre>
 * This returns a value of 0.0 at and beyond 1000 blocks distance from the origin,
 * and above it (up to 1.0) the closer you sample it to the origin.
 * <br><br>
 * See also: <code>"minecraft:y_clamped_gradient"</code> for a similar function, but for the Y coordinate only.
 */
public record Distance(double max, double min) implements DensityFunction.Base {
    public static final MapCodec<Distance> DISTANCE_CODEC =
            RecordCodecBuilder.mapCodec(
                    (instance) -> instance.group(
                                    Codec.mapEither(
                                                    Codec.doubleRange(1.0, 1000000000.0).fieldOf("max"),
                                                    Codec.doubleRange(1.0, 1000000000.0).fieldOf("argument"))

                                            .forGetter((dist) -> Either.left(dist.max())),
                                    Codec.doubleRange(0.0, 1000000000.0).optionalFieldOf("min", 0.0).forGetter(Distance::min))
                            .apply(instance, (max, min) -> {
                                if(max.left().isPresent()) {
                                    return new Distance(max.left().get(), min);
                                } else {
                                    return new Distance(max.right().orElseThrow(), min);
                                }
                            }));
    private static final CodecHolder<? extends DensityFunction> CODEC_HOLDER = CodecHolder.of(DISTANCE_CODEC);

    public Distance {
        if(max < min) {
            throw new IllegalArgumentException(String.format("Max smaller than min: %f < %f", max, min));
        }
    }

    public Distance(final double max) {
        this(max, 0.0);
    }

    @Override public CodecHolder<? extends DensityFunction> getCodecHolder() {
        return CODEC_HOLDER;
    }

    @Override public double maxValue() {
        return 1.0;
    }

    @Override public double minValue() {
        return 0.0;
    }

    @Override public double sample(NoisePos pos) {
        final double distSquared = 1.0 * pos.blockX() * pos.blockX() + 1.0 * pos.blockZ() * pos.blockZ();
        if(distSquared >= max * max) {
            /* out of maximum distance */
            return 0.0;
        } else if((min > 0.0 || max <= min) && distSquared <= min * min) {
            /* inside non-zero minimum distance; also sanity check for a "hard" border with min == max */
            return 1.0;
        } else {
            /* somewhere strictly between min and max */
            return (max - Math.sqrt(distSquared)) / (max - min);
        }
    }
}

