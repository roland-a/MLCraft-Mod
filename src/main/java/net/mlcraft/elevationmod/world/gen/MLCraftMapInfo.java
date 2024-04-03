package net.mlcraft.elevationmod.world.gen;

import net.mlcraft.elevationmod.MLCraft;
import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import net.minecraft.registry.Registerable;
import net.minecraft.registry.RegistryKey;
import net.minecraft.registry.entry.RegistryElementCodec;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.util.Identifier;

public record MLCraftMapInfo(Identifier heightmap, float horizontalScale, float verticalScale, int startingY) {
    public static final RegistryKey<MLCraftMapInfo> EMPTY = RegistryKey.of(MLCraft.MLCRAFT_INFO, MLCraft.id("empty"));
    public static void bootstrap(Registerable<MLCraftMapInfo> registerable) {
        registerable.register(EMPTY, new MLCraftMapInfo(MLCraft.id("empty"), -1, -1, Integer.MIN_VALUE));
    }
    public static final Codec<MLCraftMapInfo> CODEC = RecordCodecBuilder.create(mlcraftMapInfoInstance -> mlcraftMapInfoInstance.group(
            Identifier.CODEC.fieldOf("height_map").forGetter(MLCraftMapInfo::heightmap),
            Codec.FLOAT.fieldOf("horizontal_scale").forGetter(MLCraftMapInfo::horizontalScale),
            Codec.FLOAT.fieldOf("vertical_scale").forGetter(MLCraftMapInfo::verticalScale),
            Codec.INT.fieldOf("starting_y").forGetter(MLCraftMapInfo::startingY)
    ).apply(mlcraftMapInfoInstance, MLCraftMapInfo::new));
    public static final Codec<RegistryEntry<MLCraftMapInfo>> REGISTRY_CODEC = RegistryElementCodec.of(MLCraft.MLCRAFT_INFO, CODEC);
}