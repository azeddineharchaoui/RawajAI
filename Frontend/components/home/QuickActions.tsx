import React from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    ImageBackground,
    StyleSheet,
} from 'react-native';

type Props = {
    title: string;
    icon: number | string;
    onPress: () => void;
};

const ActionButton = ({ title, icon, onPress }: Props) => {
    return (
        <TouchableOpacity style={styles.wrapper} onPress={onPress}>
            <ImageBackground
                source={typeof icon === 'string' ? { uri: icon } : icon}
                style={styles.imageBackground}
                imageStyle={styles.image}
                resizeMode="cover"
            >
                <View style={styles.overlay}>
                    <Text style={styles.title} numberOfLines={2}>
                        {title}
                    </Text>
                </View>
            </ImageBackground>
        </TouchableOpacity>
    );
};

export default ActionButton;

const styles = StyleSheet.create({
    wrapper: {
        width: '25%',
        height: 150,
        padding: 8,
    },
    imageBackground: {
        flex: 1,
        width: '100%',
        height: '100%',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 10,
        overflow: 'hidden', // clip image inside border radius
    },
    image: {
        width: '100%',
        height: '100%',
    },
    overlay: {
        backgroundColor: 'rgba(0,0,0,0.4)',
        paddingHorizontal: 10,
        paddingVertical: 6,
        borderRadius: 6,
    },
    title: {
        color: '#fff',
        fontWeight: 'bold',
        fontSize: 14,
        textAlign: 'center',
    },
});
