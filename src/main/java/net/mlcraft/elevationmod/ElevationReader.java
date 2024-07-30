package net.mlcraft.elevationmod;

import com.mojang.serialization.Codec;
import com.mojang.serialization.MapCodec;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.util.dynamic.CodecHolder;
import net.minecraft.world.gen.densityfunction.DensityFunction;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class ElevationReader {
    public static DensityFunction.Base MC_NOISE = new DensityFunction.Base(){
        private final CodecHolder<DensityFunction> CODEC_HOLDER = CodecHolder.of(ElevationReader.CODEC);

        @Override
        public double sample(NoisePos pos) {
            if (pos.blockY() < getElevationAt(pos.blockX(), pos.blockZ())) return 1;

            return -1;
        }

        @Override
        public double minValue() {
            return -1;
        }

        @Override
        public double maxValue() {
            return 1;
        }

        @Override
        public CodecHolder<? extends DensityFunction> getCodecHolder() {
            return CODEC_HOLDER;
        }
    };

    public static final MapCodec<DensityFunction> CODEC = MapCodec.unit(MC_NOISE);

    private static byte[] elevation = null;
    private static int length;

    public static void init() throws IOException {
        var file = FabricLoader.getInstance().getConfigDir().resolve("mlcraft/elevation");

        elevation = Helper.readByteFile(file);

        length = (int)Math.sqrt(elevation.length);
    }

    private static double getElevationAt(int x, int y){
        x = Math.floorMod(x, length);
        y = Math.floorMod(y, length);

        var v = elevation[x*length+y];

        return Byte.toUnsignedInt(v) + 60;
    }

}
